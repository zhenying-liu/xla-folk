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

#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/transforms/forced_fusion.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

namespace {

bool IsMustFuseCall(const HloInstruction* caller) {
  const auto* custom_call =
      dynamic_cast<const HloCustomCallInstruction*>(caller);
  const std::string& backend_config = custom_call->raw_backend_config_string();
  // Assuming backend_config is in JSON format
  auto must_fuse_pos = backend_config.find("\"MUST_FUSE\": true");
  return must_fuse_pos != std::string::npos;
}

}  // namespace

absl::StatusOr<bool> ForcedFusion::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  auto computations = module->MakeComputationPostOrder(execution_threads);

  bool changed = false;
  unsigned count = 0;
  for (auto* computation : computations) {
    auto callers = computation->caller_instructions(HloOpcode::kCustomCall);
    for (auto* caller : callers) {
      VLOG(1) << "The custom call: " << caller->name();
      const auto* custom_call =
          dynamic_cast<const HloCustomCallInstruction*>(caller);
      if (custom_call && custom_call->custom_call_target() == "fused") {
        VLOG(1) << "The fused custom call: " << caller->name();
        TF_RET_CHECK(!IsMustFuseCall(caller))
            << "Callers with MUST_FUSE should be the only user of their "
               "computation.";

        caller->ClearCalledComputations();
        TF_RETURN_IF_ERROR(caller->parent()->ReplaceWithNewInstruction(
            caller, HloInstruction::CreateFusion(
                        caller->shape(),
                        // TODO(jreiffers): Get rid of fusion kind in XLA:GPU.
                        // It doesn't do anything useful.
                        HloInstruction::FusionKind::kLoop, caller->operands(),
                        computation)));
        VLOG(1) << "The call " << caller->name()
                << " is replaced with a fusion. "
                << "Totally " << ++count << " calls are fused by MUST_FUSE";
        changed = true;
      }
    }
  }

  return changed;
}

}  // namespace gpu
}  // namespace xla
