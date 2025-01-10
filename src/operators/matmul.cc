#include "operators/matmul.h"

namespace infini {

MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                     bool transB)
    : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}), transA(transA),
      transB(transB) {
  IT_ASSERT(checkValid(graph));
}

string MatmulObj::toString() const {
  std::ostringstream os;
  os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
     << ",A=" << inputs[0]->getGuid() << ",B=" << inputs[1]->getGuid()
     << ",C=" << outputs[0]->getGuid() << ",mnk=[" << m << "," << n << "," << k
     << "])";
  return os.str();
}

optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs) {
  // 1. The rank should match
  if (inputs.size() != 2) {
    return std::nullopt;
  }
  auto rank = inputs[0]->getRank();
  if (rank != inputs[1]->getRank()) {
    return std::nullopt;
  }
  // 2. Do transpose if needed
  Shape shapeA = inputs[0]->getDims();
  Shape shapeB = inputs[1]->getDims();
  if (transA)
    std::swap(shapeA[shapeA.size() - 1], shapeA[shapeA.size() - 2]);
  if (transB)
    std::swap(shapeB[shapeB.size() - 1], shapeB[shapeB.size() - 2]);

  // 3. Check the last two dimensions
  if (shapeA[shapeA.size() - 1] != shapeB[shapeB.size() - 2]) {
    return std::nullopt;
  }

  // 4. Calculate the output shape while checking the rest of the dimensions
  Shape outShape;
  for (size_t i = 0; i < rank - 2; ++i) {
    if (shapeA[i] == shapeB[i] || shapeA[i] == 1 || shapeB[i] == 1) {
      outShape.push_back(std::max(shapeA[i], shapeB[i]));
    } else {
      return std::nullopt;
    }
  }

  // 5. Append the last two dimensions
  outShape.push_back(shapeA[shapeA.size() - 2]);
  outShape.push_back(shapeB[shapeB.size() - 1]);

  return {{outShape}};
}

} // namespace infini