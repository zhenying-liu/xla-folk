/* Copyright 2016 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/fused_out_spaces.h"

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

using FusedOutSpacesTest = HloHardwareIndependentTestBase;

TEST_F(FusedOutSpacesTest, BasicTest) {
  constexpr absl::string_view kHloString = R"(
HloModule jit_foo, entry_computation_layout={(f32[3]{0:S(5)}, f32[3]{0})->(f32[3]{0:S(5)}, f32[3]{0})}

%fused.8 (Arg_0.3: f32[3], Arg_1.4: f32[3]) -> (f32[3], f32[3]) {
  %Arg_0.3 = f32[3]{0} parameter(0)
  %Arg_1.4 = f32[3]{0} parameter(1)
  %add.5 = f32[3]{0} add(%Arg_0.3, %Arg_1.4)
  %mul.6 = f32[3]{0} multiply(%Arg_0.3, %Arg_1.4)
  ROOT %tuple.7 = (f32[3]{0}, f32[3]{0}) tuple(%add.5, %mul.6)
}

ENTRY %main.13 (x.1: f32[3], y.2: f32[3]) -> (f32[3], f32[3]) {
  %x.1 = f32[3]{0:S(5)} parameter(0)
  %y.2 = f32[3]{0} parameter(1)
  ROOT %fused_call.9 = (f32[3]{0}, f32[3]{0}) custom-call(%x.1, %y.2), custom_call_target="fused", api_version=API_VERSION_TYPED_FFI, called_computations={%fused.8}, backend_config={MUST_FUSE = true, inlineable = false, out_spaces = ["Space.Host", "Space.Device"]}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));
  VLOG(3) << module->ToString();
  bool changed;
  TF_ASSERT_OK_AND_ASSIGN(changed, FusedOutSpaces().Run(module.get()));
  EXPECT_TRUE(changed);
  VLOG(3) << "After running the FusedOutSpaces pass: ";
  VLOG(3) << module->ToString();
  // auto module = std::move(module_status).value();

  // auto run_status = FusedOutSpaces().Run(module.get());
  // TF_ASSERT_OK(run_status.status());
  // EXPECT_TRUE(run_status.value());

  // VLOG(3) << module->ToString();
  // TF_ASSERT_OK(verifier().Run(module.get()).status());
}

}  // namespace
}  // namespace gpu
}  // namespace xla
