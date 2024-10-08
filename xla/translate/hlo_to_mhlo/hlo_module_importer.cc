/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/translate/hlo_to_mhlo/hlo_module_importer.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/layout.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/shape_layout.h"
#include "xla/shape_util.h"
#include "xla/translate/hlo_to_mhlo/hlo_function_importer.h"
#include "xla/translate/hlo_to_mhlo/hlo_utils.h"
#include "xla/translate/hlo_to_mhlo/module_config_importer.h"
#include "xla/xla.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {

HloModuleImporter::HloModuleImporter(mlir::ModuleOp module,
                                     bool import_all_computation,
                                     bool flatten_computation_args_result)
    : import_all_computation_(import_all_computation),
      flatten_computation_args_result_(flatten_computation_args_result),
      symbol_table_(module),
      builder_(module.getContext()) {
  module.getContext()->loadDialect<mlir::arith::ArithDialect>();
  module.getContext()->loadDialect<mlir::func::FuncDialect>();
  module.getContext()->loadDialect<mlir::mhlo::MhloDialect>();
  module.getContext()->loadDialect<mlir::quant::QuantizationDialect>();
}

namespace {

constexpr char kFrontendAttributesAttr[] = "mhlo.frontend_attributes";

mlir::ArrayAttr ConvertCrossProgramPrefetches(
    const absl::Span<const HloModule::CrossProgramPrefetchInfo> prefetches,
    const HloComputation& entryComputation, mlir::Builder* builder,
    bool flatten_computation_args_result) {
  llvm::SmallVector<mlir::Attribute, 4> shapes;
  shapes.reserve(prefetches.size());
  if (flatten_computation_args_result) {
    llvm::SmallVector<absl::flat_hash_map<ShapeIndex, int64_t>>
        original_param_index_to_flattened_arg_index;
    int64_t arg_index = 0;
    for (HloInstruction* param_instruction :
         entryComputation.parameter_instructions()) {
      auto& param_map =
          original_param_index_to_flattened_arg_index.emplace_back();
      ShapeUtil::ForEachLeafShape(param_instruction->shape(),
                                  [&](const Shape&, const ShapeIndex& index) {
                                    param_map[index] = arg_index++;
                                  });
    }
    for (const auto& [parameter, index, alt_memory_offset] : prefetches) {
      shapes.push_back(mlir::mhlo::CrossProgramPrefetchAttr::get(
          builder->getContext(),
          original_param_index_to_flattened_arg_index[parameter][index],
          /*indices=*/{}, alt_memory_offset));
    }
  } else {
    for (const auto& [parameter, index, alt_memory_offset] : prefetches) {
      shapes.push_back(mlir::mhlo::CrossProgramPrefetchAttr::get(
          builder->getContext(), parameter,
          llvm::ArrayRef<int64_t>(index.data(), index.size()),
          alt_memory_offset));
    }
  }

  return mlir::ArrayAttr::get(builder->getContext(), shapes);
}
}  // namespace

absl::Status HloModuleImporter::Import(const HloModule& hlo_module) {
  auto module = llvm::cast<mlir::ModuleOp>(symbol_table_.getOp());
  module.setName(hlo_module.name());
  module->setAttr(
      "mhlo.cross_program_prefetches",
      ConvertCrossProgramPrefetches(hlo_module.CrossProgramPrefetches(),
                                    *hlo_module.entry_computation(), &builder_,
                                    flatten_computation_args_result_));
  module->setAttr(
      "mhlo.is_dynamic",
      mlir::BoolAttr::get(builder_.getContext(), hlo_module.is_dynamic()));
  ImportFrontendAttributes(hlo_module, module);
  ImportHloModuleConfig(hlo_module.config(), module);
  module->setAttr("mhlo.use_auto_spmd_partitioning",
                  mlir::BoolAttr::get(builder_.getContext(),
                                      hlo_module.use_auto_spmd_partitioning()));
  if (hlo_module.has_spmd_output_sharding()) {
    module->setAttr(
        "mhlo.spmd_output_sharding",
        ConvertSharding(hlo_module.spmd_output_sharding(), &builder_));
  }

  module->setAttr("mhlo.input_output_alias",
                  ConvertInputOutputAlias(
                      hlo_module.input_output_alias_config(), &builder_));

  if (hlo_module.has_spmd_parameters_shardings()) {
    llvm::SmallVector<mlir::Attribute> parameter_shardings;
    parameter_shardings.reserve(hlo_module.spmd_parameters_shardings().size());
    for (const auto& root_sharding : hlo_module.spmd_parameters_shardings()) {
      llvm::ArrayRef<HloSharding> shardings = root_sharding;
      if (root_sharding.IsTuple() && flatten_computation_args_result_) {
        shardings = root_sharding.tuple_elements();
      }
      for (const auto& sharding : shardings) {
        parameter_shardings.push_back(ConvertSharding(sharding, &builder_));
      }
    }
    module->setAttr("mhlo.spmd_parameters_shardings",
                    builder_.getArrayAttr(parameter_shardings));
  }

  if (!import_all_computation_)
    // Only import the entry computation, any reachable one will be imported
    // unless turned into a region operation.
    return HloFunctionImporter::ImportAsFunc(
               *hlo_module.entry_computation(), symbol_table_, &function_map_,
               &builder_,
               /*is_main*/ true, flatten_computation_args_result_)
        .status();

  // The MLIR CPU pipeline assumes default layouts throughout the program. At
  // the boundaries, this may not be the case, so layout information needs to
  // be propagated to adapt the data layouts.
  if (const auto& computation_layout = hlo_module.entry_computation_layout();
      computation_layout.LayoutIsSet()) {
    if (HasCustomLayout(computation_layout.result_layout().shape())) {
      if (computation_layout.result_layout().shape().IsTuple()) {
        llvm::SmallVector<mlir::Attribute> result_layouts;
        llvm::SmallVector<mlir::Attribute> result_tiles;
        for (auto& tuple_element_layout :
             computation_layout.result_layout().shape().tuple_shapes()) {
          std::pair<mlir::Attribute, mlir::Attribute> layout_attrs =
              GetLayoutAttribute(builder_, tuple_element_layout);
          result_layouts.push_back(layout_attrs.first);
          result_tiles.push_back(layout_attrs.second);
        }
        module->setAttr(
            "mhlo.xla_entry_computation_result_layout",
            builder_.getArrayAttr({builder_.getArrayAttr(result_layouts)}));
        module->setAttr(
            "mhlo.xla_entry_computation_result_tiles",
            builder_.getArrayAttr({builder_.getArrayAttr(result_tiles)}));
      } else {
        std::pair<mlir::Attribute, mlir::ArrayAttr> layout_attrs =
            GetLayoutAttribute(builder_,
                               computation_layout.result_layout().shape(),
                               computation_layout.result_layout().layout());
        module->setAttr("mhlo.xla_entry_computation_result_layout",
                        builder_.getArrayAttr({layout_attrs.first}));
        module->setAttr("mhlo.xla_entry_computation_result_tiles",
                        builder_.getArrayAttr({layout_attrs.second}));
      }
    }
    if (llvm::any_of(computation_layout.parameter_layouts(),
                     [](const ShapeLayout& shape) {
                       return HasCustomLayout(shape.shape());
                     })) {
      llvm::SmallVector<mlir::Attribute> parameter_layouts;
      llvm::SmallVector<mlir::Attribute> parameter_tiles;
      for (auto& layout : computation_layout.parameter_layouts()) {
        if (layout.shape().IsTuple()) {
          llvm::SmallVector<mlir::Attribute> tuple_element_parameter_layouts;
          llvm::SmallVector<mlir::Attribute> tuple_element_parameter_tiles;
          for (auto& tuple_element_shape : layout.shape().tuple_shapes()) {
            std::pair<mlir::Attribute, mlir::Attribute> layout_attrs =
                GetLayoutAttribute(builder_, tuple_element_shape);
            tuple_element_parameter_layouts.push_back(layout_attrs.first);
            tuple_element_parameter_tiles.push_back(layout_attrs.second);
          }
          parameter_layouts.push_back(
              builder_.getArrayAttr({tuple_element_parameter_layouts}));
          parameter_tiles.push_back(
              builder_.getArrayAttr({tuple_element_parameter_tiles}));
        } else {
          std::pair<mlir::Attribute, mlir::ArrayAttr> layout_attrs =
              GetLayoutAttribute(builder_, layout.shape());
          parameter_layouts.push_back(layout_attrs.first);
          parameter_tiles.push_back(layout_attrs.second);
        }
      }
      module->setAttr("mhlo.xla_entry_computation_parameter_layouts",
                      builder_.getArrayAttr({parameter_layouts}));
      module->setAttr("mhlo.xla_entry_computation_parameter_tiles",
                      builder_.getArrayAttr({parameter_tiles}));
    }
  }

  auto* module_entry_computation = hlo_module.entry_computation();
  for (const auto* computation : hlo_module.computations())
    TF_RETURN_IF_ERROR(HloFunctionImporter::ImportAsFunc(
                           *computation, symbol_table_, &function_map_,
                           &builder_,
                           /*is_main*/ computation == module_entry_computation,
                           flatten_computation_args_result_)
                           .status());

  return absl::OkStatus();
}

absl::Status HloModuleImporter::Import(const HloModuleProto& module_proto) {
  DebugOptions debug_options;
  TF_ASSIGN_OR_RETURN(
      auto module_config,
      HloModule::CreateModuleConfigFromProto(module_proto, debug_options));
  TF_ASSIGN_OR_RETURN(auto module,
                      HloModule::CreateFromProto(module_proto, module_config));

  return Import(*module);
}

void HloModuleImporter::ImportFrontendAttributes(const HloModule& hlo_module,
                                                 mlir::ModuleOp module) {
  if (!hlo_module.frontend_attributes().map().empty()) {
    llvm::SmallVector<mlir::NamedAttribute, 4> frontend_attributes;
    for (const auto& [k, v] : hlo_module.frontend_attributes().map()) {
      frontend_attributes.push_back(
          builder_.getNamedAttr(k, builder_.getStringAttr(v)));
    }
    if (!frontend_attributes.empty()) {
      module->setAttr(kFrontendAttributesAttr,
                      builder_.getDictionaryAttr(frontend_attributes));
    }
  }
}

}  // namespace xla
