#include "operators/concat.h"
#include "utils/operator_utils.h"

namespace infini {
ConcatObj::ConcatObj(GraphObj *graph, TensorVec inputs, Tensor output, int _dim)
    : OperatorObj(OpType::Concat, inputs, {output}) {
  int rank = inputs[0]->getRank();
  dim = get_real_axis(_dim, rank);
  IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> ConcatObj::inferShape(const TensorVec &inputs) {
  Shape dims = inputs[0]->getDims();
  auto rank = inputs[0]->getRank();

  dims[dim] = 0;
  for (const auto &input : inputs) {
    IT_ASSERT(input->getRank() == rank);
    for (size_t i = 0; i < rank; i++) {
      if (i == static_cast<size_t>(dim)) {
        dims[i] += input->getDims()[i];
      } else {
        IT_ASSERT(dims[i] == input->getDims()[i]);
      }
    }
  }

  return {{dims}};
}

std::string ConcatObj::toString() const {
  std::ostringstream os;
  os << "Concat[" << getGuid() << "]";
  os << "(";
  for (auto input : inputs)
    os << vecToString(input->getDims()) << ",";
  os << "dim=" << dim << ",";
  os << "input=";
  for (auto input : inputs)
    os << input->getGuid() << ",";
  os << "output=" << outputs[0]->getGuid() << ")";
  return os.str();
}

} // namespace infini
