/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include <algorithm>
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/layout_util.h"
#include "xla/shape_util.h"
#include "xla/service/gpu/transforms/fused_out_spaces.h"
#include "xla/shape_util.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

absl::StatusOr<bool> HasOutMemorySpaces(
    const HloInstruction* caller,
    absl::InlinedVector<int64_t, 2>& memory_spaces) {
  const auto* custom_call =
      dynamic_cast<const HloCustomCallInstruction*>(caller);
  if (!custom_call) {
    return false;
  }
  const std::string& backend_config = custom_call->raw_backend_config_string();

  auto out_spaces_pos = backend_config.find("out_spaces");
  if (out_spaces_pos != std::string::npos) {
    auto start_pos = backend_config.find('[', out_spaces_pos);
    auto end_pos = backend_config.find(']', start_pos);
    if (start_pos != std::string::npos && end_pos != std::string::npos) {
      std::string spaces_str =
          backend_config.substr(start_pos + 1, end_pos - start_pos - 1);
      std::vector<std::string> out_spaces;
      size_t pos = 0;
      while ((pos = spaces_str.find('"')) != std::string::npos) {
        size_t end_quote = spaces_str.find('"', pos + 1);
        if (end_quote == std::string::npos) {
          break;
        }
        out_spaces.push_back(spaces_str.substr(pos + 1, end_quote - pos - 1));
        spaces_str = spaces_str.substr(end_quote + 1);
      }
      for (const auto& space : out_spaces) {
        if (space == "Space.Host") {
          memory_spaces.push_back(Layout::kHostMemorySpace);
        } else if (space == "Space.Device" || space == "None") {
          memory_spaces.push_back(Layout::kDefaultMemorySpace);
        } else {
          // Handle unknown space
          return absl::InternalError(
              absl::StrCat("Unknown memory space: ", space));
        }
      }
      return true;
    }
  }
  return false;
}

void UpdateMemorySpaces(Shape* shape,
                        const absl::InlinedVector<int64_t, 2>& memory_spaces) {
  if (memory_spaces.empty()) {
    return;
  }

  ShapeUtil::ForEachMutableSubshape(
      shape, [&](Shape* subshape, const ShapeIndex& index) {
        if (!subshape->has_layout() || ShapeUtil::IsScalar(*subshape)) {
          return;
        }

        int64_t memory_space_index = 0;

        if (shape->IsTuple() && index.size() > 0) {
          // For tuple shapes, use the first index to determine which output
          // this is
          memory_space_index =
              std::min(static_cast<int64_t>(index[0]),
                       static_cast<int64_t>(memory_spaces.size() - 1));
        } else if (!shape->IsTuple()) {
          // For single return shapes, use the first (and only) memory space
          memory_space_index = 0;
        }

        int64_t memory_space = memory_spaces[memory_space_index];

        // ONLY set memory space, preserve existing layout dimensions
        subshape->mutable_layout()->set_memory_space(memory_space);
      });
}

void UpdateMemorySpaceForInputs(HloInstruction* caller) {
  auto* fused_computation = caller->called_computations()[0];
  for (int64_t i = 0; i < caller->operand_count(); ++i) {
    HloInstruction* operand = caller->mutable_operand(i);
    HloInstruction* parameter = fused_computation->parameter_instruction(i);

    // Only update memory space, preserve existing layout dimensions
    if (!ShapeUtil::IsScalar(operand->shape()) &&
        operand->shape().has_layout() && parameter->shape().has_layout()) {
      int64_t operand_memory_space = operand->shape().layout().memory_space();
      parameter->mutable_shape()->mutable_layout()->set_memory_space(
          operand_memory_space);
    }
  }
}

void BackpropagateMemorySpaces(HloInstruction* instruction) {
  if (instruction->shape().IsTuple()) {
    const Shape& tuple_shape = instruction->shape();
    for (int64_t i = 0; i < instruction->operand_count(); ++i) {
      HloInstruction* operand = instruction->mutable_operand(i);
      const Shape& target_subshape = ShapeUtil::GetSubshape(tuple_shape, {i});

      // Only update memory space, preserve existing layout dimensions
      if (operand->shape().has_layout() && target_subshape.has_layout() &&
          !ShapeUtil::IsScalar(operand->shape())) {
        int64_t target_memory_space = target_subshape.layout().memory_space();
        operand->mutable_shape()->mutable_layout()->set_memory_space(
            target_memory_space);
      }
    }
  }
}

absl::StatusOr<bool> FusedOutSpaces::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  auto computations = module->MakeComputationPostOrder(execution_threads);

  bool changed = false;

  for (auto* computation : computations) {
    VLOG(1) << "Current computation is: " << computation->ToString();
    auto callers = computation->caller_instructions(HloOpcode::kCustomCall);
    for (auto* caller : callers) {
      VLOG(1) << "The custom call: " << caller->name();
      const auto* custom_call =
          dynamic_cast<const HloCustomCallInstruction*>(caller);
      if (custom_call && custom_call->custom_call_target() == "fused") {
        VLOG(1) << "The fused custom call: " << caller->name();

        absl::InlinedVector<int64_t, 2> memory_spaces;
        TF_ASSIGN_OR_RETURN(bool has_out_spaces,
                            HasOutMemorySpaces(caller, memory_spaces));
        if (has_out_spaces) {
          VLOG(1) << "Memory spaces: " << absl::StrJoin(memory_spaces, ", ");

          // Update memory spaces for the fused computation's root instruction
          auto* fused_computation = custom_call->called_computations()[0];
          auto* root_instruction = fused_computation->root_instruction();
          Shape new_shape = root_instruction->shape();
          UpdateMemorySpaces(&new_shape, memory_spaces);
          *root_instruction->mutable_shape() = new_shape;

          // Backpropagate memory spaces from the root instruction if it's a
          // tuple
          BackpropagateMemorySpaces(root_instruction);

          // Update memory spaces for the custom call's output shapes
          Shape caller_shape = caller->shape();
          UpdateMemorySpaces(&caller_shape, memory_spaces);
          *caller->mutable_shape() = caller_shape;

          // Update memory spaces for the custom call's input shapes
          UpdateMemorySpaceForInputs(caller);

          // Update memory spaces for the get-tuple-element instructions
          for (auto* user : caller->users()) {
            if (user->opcode() == HloOpcode::kGetTupleElement) {
              int64_t index = user->tuple_index();
              const Shape& target_subshape =
                  ShapeUtil::GetSubshape(caller_shape, {index});

              // Only update memory space, preserve existing layout dimensions
              if (user->shape().has_layout() && target_subshape.has_layout() &&
                  !ShapeUtil::IsScalar(user->shape())) {
                int64_t target_memory_space =
                    target_subshape.layout().memory_space();
                user->mutable_shape()->mutable_layout()->set_memory_space(
                    target_memory_space);
              }

              // Update the tuple instruction shape if necessary
              for (auto* tuple_user : user->users()) {
                if (tuple_user->opcode() == HloOpcode::kTuple) {
                  // Create new tuple shape with updated memory spaces but
                  // preserved layout dimensions
                  Shape new_tuple_shape = tuple_user->shape();
                  absl::InlinedVector<int64_t, 2> tuple_memory_spaces;
                  for (const HloInstruction* operand : tuple_user->operands()) {
                    if (operand->shape().has_layout()) {
                      tuple_memory_spaces.push_back(
                          operand->shape().layout().memory_space());
                    } else {
                      tuple_memory_spaces.push_back(
                          Layout::kDefaultMemorySpace);
                    }
                  }
                  // Only update memory spaces, preserve layout dimensions
                  UpdateMemorySpaces(&new_tuple_shape, tuple_memory_spaces);
                  *tuple_user->mutable_shape() = new_tuple_shape;
                }
              }
            }
          }

          changed = true;
        } else {
          TF_RET_CHECK(false) << "Callers with MUST_FUSE should be the only "
                                 "user of their computation.";
        }
      }
    }
  }

  return changed;
}

}  // namespace gpu
}  // namespace xla
