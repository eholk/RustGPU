use OpenCL::vector::VectorType;

enum IntOrFloat {
    Int(int), Float(float)
}

impl IntOrFloat: VectorType;

enum EvenOdd {
    Unknown, Even, Odd
}

impl EvenOdd: VectorType;
