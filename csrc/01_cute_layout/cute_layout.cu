#include <cute/layout.hpp>
#include <iostream>

#include <cute/layout_composed.hpp>  // cute::composition

using namespace cute;

void LearnMakeLayout() {
  auto a = Layout<_1, _0>{};
}


// Reference: https://zhuanlan.zhihu.com/p/28356098779
void LearnLayoutComposition() {
  /* 
  * A o B (c)
  *   Recall that layout maps a (logical) index to a memory location (1D)
  *   composition means to use the outputs of B as new indices for A
  *   the result would be a subset of A's output, somehow rearranged
  * 
  * Let's go through some examples first
  * ========== one dimensional ==========
  * A's shape: 10, stride: 4
  *   index: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  *   value: [0, 4, 8, 12, 16, 20, 24, 28, 32, 36]
  * B's shape: 3, stride: 2
  *   index: [0, 1, 2]
  *   value: [0, 2, 4]
  * // treat the output of B as a new index for A
  * so A o B's shape: 3, stride: 8
  *   index: [0, 1, 2]
  *   value: [0, 8, 16]
  */
  {
    std::cout << "========== one dimensional ==========" << std::endl;
    auto a = Layout<_10, _4>{};
    auto b = Layout<_3, _2>{};
    auto result = composition(a, b);
    std::cout << "a: " << a << ", b: " << b << ", result: " << result << std::endl;
    // Same as above, different syntax
    // result = a.compose(b);
    // std::cout << "a: " << a << ", b: " << b << ", result: " << result << std::endl;
  }
  /*
  * ========== two dimensional ==========
  * A: (3,4):(4,1)
  *   index: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
  *   coordinates: [(0,0) (1,0) (2,0) (0,1) (1,1) (2,1) (0,2) (1,2) (2,2) (0,3) (1,3) (2,3)]
  *   value: [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
  * B: 3:2
  *   index: [0, 1, 2]
  *   value: [0, 2, 4]
  * // treat the output of B as a new index for A
  * // select the elements at the positions specified by B
  * so A o B' is
  *   index: [0, 1, 2]
  *   value: [0, 8, 5]
  * // inconsistent stride, compile failed
  * {
  *   auto a = Layout<Shape<_3, _4>, Stride<_4, _1>>{};
  *   auto b = Layout<_3, _2>{};
  *   auto result = composition(a, b);
  *   std::cout << "a: " << a << ", b: " << b << ", result: " << result << std::endl;
  *   // static_assert(IntTupleA::value % IntTupleB::value == 0 || IntTupleB::value % IntTupleA::value == 0, "Static shape_div failure");
  * }
  * 
  * Let's try another example
  * A: (4,3):(3,1)
  *   index: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
  *   coordinates: [(0,0) (1,0) (2,0) (3,0) (0,1) (1,1) (2,1) (3,1) (0,2) (1,2) (2,2) (3,2)]
  *   value: [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
  * B: 12:2
  *   index: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
  *   value: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
  * A o B'：
  *   B's index: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
  *   B's value (A's index): [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
  *   A o B's value: [0, 6, 1, 7, 2, 8]
  * How to determine the shape of the result?
  * What about the B's index which is not in the range of A's index?
  */
  {
    std::cout << "========== two dimensional ==========" << std::endl;
    auto a = Layout<Shape<_4, _3>, Stride<_3, _1>>{};
    auto b = Layout<_12, _2>{};
    auto result = composition(a, b);
    std::cout << "a: " << a << ", b: " << b << ", result: " << result << std::endl;
  }
  /*
  * ========== two dimensional x two dimensional ==========
  * A: (4,3):(3,1)
  *   index: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
  *   coordinates: [(0,0) (1,0) (2,0) (3,0) (0,1) (1,1) (2,1) (3,1) (0,2) (1,2) (2,2) (3,2)]
  *   value: [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
  * B: (2,2):(2,1)
  *   index: [0, 1, 2, 3]
  *   coordinates: [(0,0) (1,0) (0,1) (1,1)]
  *   value: [0, 2, 1, 3]
  * // Flatten B
  * A o B:
  *   B's index: [0, 1, 2, 3]
  *   B's value (A's index): [0, 2, 1, 3]
  *   A o B's value: [0, 6, 3, 9]
  * // Reshape to B's shape
  */
  {
    std::cout << "========== two dimensional x two dimensional ==========" << std::endl;
    auto a = Layout<Shape<_4, _3>, Stride<_3, _1>>{};
    auto b = Layout<Shape<_2, _2>, Stride<_2, _1>>{};
    auto result = composition(a, b);
    std::cout << "a: " << a << ", b: " << b << ", result: " << result << std::endl;
  }
  /*
  * ========== the rules of composition ==========
  * non-modal: layout without brackets
  * case 1: non-modal A o non-modal B
  * // Along the indices of A, select (the shape of B) elements, strided by the stride of B
  * A: 8:3
  * B: 4:2 
  * A o B: 4:3*2 = 4:6
  */
  {
    std::cout << "========== case 1: non-modal A o non-modal B ==========" << std::endl;
    auto a = Layout<_8, _3>{};
    auto b = Layout<_4, _2>{};
    auto result = composition(a, b);
    std::cout << "a: " << a << ", b: " << b << ", result: " << result << std::endl;
  }
  /*
  * case 2: non-modal A o modal B
  * A: 20:2
  * B: (5,4):(4,1)
  * A o B = A o {B1, B2} = {A o B1, A o B2}
  * // Split B into two non-modal layouts
  * A o B1 = 5:2*4 = 5:8
  * A o B2 = 4:2*1 = 4:2
  * A o B = (5,4):(8,2)
  */
  {
    std::cout << "========== case 2: non-modal A o modal B ==========" << std::endl;
    auto a = Layout<Int<20>, _2>{};
    auto b = Layout<Shape<_5, _4>, Stride<_4, _1>>{};
    auto result = composition(a, b);
    std::cout << "a: " << a << ", b: " << b << ", result: " << result << std::endl;
  }
  /*
  * case 3: modal A o non-modal B
  * case 3.1 B's stride is 1 -> `Mod_Out`
  * case 3.2 B's stride is not 1 -> `Div_Out`, then `Mod_Out`
  * A: (3,6,2,8):(96,16,8,1)
  * B: 16:9
  * 
  * firstly: `Div_Out`
  * `Div_Out` means to select one element from A from N elements
  *   input: Layout A and stride B
  *   output: new Layout C
  *   i = 0:
  *     ShapeA[0] = 3
  *     StrideA[0] = 96
  *     stride: 9
  *     ShapeC[0] = ceil_div(ShapeA[0], Stride) = ceil_div(3,9) = 1
  *     // Stride for the next iteration
  *     Stride = ceil_div(Stride, ShapeA[0]) = ceil_div(9,3) = 3
  *     StrideC[0] = # // Does not matter, while the ShapeC[0] is 1
  *     ShapeC = (1,?,?,?)
  *     StrideC = (#,?,?,?)
  *   i = 1:
  *     ShapeA[1] = 6
  *     StrideA[1] = 16
  *     stride: 3
  *     ShapeC[1] = ceil_div(ShapeA[1], Stride)	= ceil_div(6,3) = 2
  *     // Stride for the next iteration
  *     Stride = ceil_div(Stride, ShapeA[1]) = ceil_div(3,6) = 1
  *     StrideC[1] = Stride * StrideA[1] = 3 * 16 = 48
  *     ShapeC = (1,2,?,?)
  *     StrideC = (#,48,?,?)
  *     // The 0th and 1st iterations are easy to understand, we need 9 elements, but the 0th axis of A has 3 elements,
  *     // 0th and 1st axes of A have 18 elements, so the result shape would be 2
  *     // We need to select 1 elements every 3 elements from the 1st axis, so on top of the StrideA[1], we need to multiply 3
  *     //
  *     // Stride is 1, `Div_Out` terminates here
  *     ShapeC = (1,2,2,8)
  *     StrideC = (#,48,8,1)
  * 
  * ShapeC = (2,2,8)
  * secondly: `Mod_Out`
  * `Mod_Out` means to select total M elements
  *   input: ShapeC and ShapeB
  *   output: ShapeD // StrideD is the same as StrideC
  *   i = 0:
  *     ShapeC[0] = 2
  *     ShapeB = 16
  *     ShapeD[0] = min(ShapeC[0], Shape) = min(2,16) = 2
  *     // Shape for the next iteration
  *     Shape = ceil_div(Shape, ShapeC[0])= ceil_div(16,2) = 8
  *     ShapeD = (2,?,?)
  *   i = 1:
  *     ShapeC[1] = 2
  *     ShapeB = 8
  *     ShapeD[1] = min(ShapeC[1], Shape) = min(2,8) = 2
  *     // Shape for the next iteration
  *     Shape = ceil_div(Shape, ShapeC[1])= ceil_div(8,2) = 4
  *     ShapeD = (2,2,?)
  *   i = 2:
  *     ShapeC[2] = 8
  *     ShapeB = 4
  *     ShapeD[2] = min(ShapeC[2], Shape) = min(8,4) = 4
  *     // Shape for the next iteration
  *     Shape = ceil_div(Shape, ShapeC[2])= ceil_div(4,8) = 1
  *     ShapeD = (2,2,4)
  *     Shape is 1, `Mod_Out` terminates here
  * 
  * A o B = (2,2,4):(48,8,1)
  */
  {
    std::cout << "========== case 3: modal A o non-modal B ==========" << std::endl;
    auto a = Layout<Shape<_3, _6, _2, _8>, Stride<_96, _16, _8, _1>>{};
    auto b = Layout<_16, _9>{};
    auto result = composition(a, b);
    std::cout << "a: " << a << ", b: " << b << ", result: " << result << std::endl;
  }
  /*
  * case 4: modal A o modal B
  * A o B = A o {B1, B2} = {A o B1, A o B2}
  * 
  * Let's skip this case for now...
  */

  /*
  * Back to our two dimensional example
  * A: (4,3):(3,1)
  * B: 12:2
  * 
  * `Div_Out`
  *   i = 0:
  *     ShapeA[0] = 4
  *     StrideA[0] = 3
  *     Stride: 2
  *     ShapeC[0] = ceil_div(ShapeA[0], Stride) = ceil_div(4,2) = 2
  *     Stride = ceil_div(Stride, ShapeA[0]) = ceil_div(2,4) = 1
  *     StrideC[0] = StrideA[0] * Stride = 3 * 2 = 6
  *     ShapeC = (2,?)
  *     StrideC = (6,?)
  *     // Stride is 1, `Div_Out` terminates here
  * 
  *  ShapeC = (2,3)
  *  StrideC = (6,1)
  * 
  * `Mod_Out`
  *   i = 0:
  *     ShapeC[0] = 2
  *     Shape = 12
  *     ShapeD[0] = min(ShapeC[0], Shape) = min(2,12) = 2
  *     Shape = ceil_div(Shape, ShapeC[0]) = ceil_div(12,2) = 6
  *     ShapeD = (2,?)
  *   i = 1:
  *     ShapeC[1] = 3
  *     Shape = 6
  *     ShapeD[1] = min(ShapeC[1], Shape) = min(3,6) = 3
  *     Shape = ceil_div(Shape, ShapeC[1]) = ceil_div(6,3) = 2
  *     ShapeD = (2,2)
  *   // This is where I get stuck, there's not enough elements to select from A
  */
  {
    std::cout << "========== two dimensional (even larger B) ==========" << std::endl;
    auto a = Layout<Shape<_4, _3>, Stride<_3, _1>>{};
    auto b = Layout<_24, _2>{};
    auto result = composition(a, b);
    std::cout << "a: " << a << ", b: " << b << ", result: " << result << std::endl;
  }
  // It seems that Layout A will try to fit B
}

int main() {
  LearnMakeLayout();
  LearnLayoutComposition();
  std::cout << "done" << std::endl;
  return 0;
}
