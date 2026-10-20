#include "mie/engine/inference_request.h"

namespace mie {

Status PrepareRequest(InferenceRequest* request) {
  if (request->inputs.empty()) return InvalidArgument("request has no inputs");

  int64_t batch = -1;
  std::string signature;
  for (const auto& [name, tensor] : request->inputs) {
    MIE_RETURN_IF_ERROR(tensor.Validate());
    if (tensor.shape().empty()) return InvalidArgument("input '" + name + "' must have a batch dimension");
    const int64_t b = tensor.shape()[0];
    if (b <= 0) return InvalidArgument("input '" + name + "' has an empty batch dimension");
    if (batch >= 0 && b != batch) {
      return InvalidArgument("inputs disagree on batch size (" + std::to_string(batch) + " vs " +
                             std::to_string(b) + " for '" + name + "')");
    }
    batch = b;

    // Signature = names + dtypes + non-batch dims. TensorMap is ordered, so the
    // signature is canonical.
    signature += name;
    signature += ':';
    signature += DataTypeName(tensor.dtype());
    for (size_t i = 1; i < tensor.shape().size(); ++i) {
      signature += 'x';
      signature += std::to_string(tensor.shape()[i]);
    }
    signature += ';';
  }
  request->batch_size = batch;
  request->batch_signature = std::move(signature);
  return Status::Ok();
}

}  // namespace mie
