#ifndef RANDOM_WARP_AFFINE_H_
#define RANDOM_WARP_AFFINE_H_

#include "dali/pipeline/operators/operator.h"

namespace custom_ops {

template <typename Backend>
class RandomWarpAffine : public ::dali::Operator<Backend> {
 public:
  inline explicit RandomWarpAffine(const ::dali::OpSpec &spec) :
    ::dali::Operator<Backend>(spec) {
    
         angle_ = spec.GetArgument<float>("angle");
         angle_ = spec.GetArgument<float>("angle");         
    }

  virtual inline ~RandomWarpAffine() = default;

  RandomWarpAffine(const RandomWarpAffine&) = delete;
  RandomWarpAffine& operator=(const RandomWarpAffine&) = delete;
  RandomWarpAffine(RandomWarpAffine&&) = delete;
  RandomWarpAffine& operator=(RandomWarpAffine&&) = delete;

 protected:
  void RunImpl(::dali::Workspace<Backend> *ws, const int idx) override;
};

}  // namespace other_ns

#endif  // EXAMPLE_DUMMY_H_