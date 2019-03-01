#include "random_warp_affine.h"

namespace custom_ops {

template<>
void RandomWarpAffine<::dali::CPUBackend>::RunImpl(::dali::SampleWorkspace *ws, const int idx) {
  auto &input = ws->Input<::dali::CPUBackend>(idx);
  auto output = ws->Output<::dali::CPUBackend>(idx);
  output->set_type(input.type());
  output->ResizeLike(input);

  ::dali::TypeInfo type = input.type();
  type.Copy<::dali::CPUBackend, ::dali::CPUBackend>(
      output->raw_mutable_data(),
      input.raw_data(), input.size(), 0);
}

}  // namespace other_ns

DALI_REGISTER_OPERATOR(RandomWarpAffine, ::custom_ops::RandomWarpAffine<::dali::CPUBackend>, ::dali::CPU);

DALI_SCHEMA(RandomWarpAffine)
  .DocStr("Random warp affine transformation")
  .NumInput(1)
  .NumOutput(1);