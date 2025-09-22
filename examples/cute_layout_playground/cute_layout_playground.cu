#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>

#include <cute/layout.hpp>
#include <cute/swizzle.hpp>
#include <cute/tensor.hpp>

#include <cutlass/util/command_line.h>

int main(int argc, const char** argv)
{
    cute::Step<cute::_1, cute::X, cute::_1> projection{cute::Step<cute::_1, cute::X, cute::_1>{}};
    cute::Layout<cute::Shape<cute::_2,cute::_16,cute::_1>, cute::Stride<cute::_16,cute::_1,cute::_0>> thr_layout{};
    auto tile{cute::dice(projection, thr_layout)};
    std::cout << "Projection: " << std::endl;
    cute::print(projection);
    std::cout << std::endl;
    std::cout << "Thread Layout: " << std::endl;
    cute::print(thr_layout);
    std::cout << std::endl;
    std::cout << "Tile: " << std::endl;
    cute::print(tile);
    std::cout << std::endl;

    // Create a tensor of shape [M, K]
    // M is a static 128, K is a static 16
    using M = cute::Int<128>;
    using K = cute::Int<16>;
    cute::Tensor tensor{cute::make_tensor<float>(cute::Shape<M, K>{})};
    int thr_idx = 4;
    cute::Tensor thrA = local_partition(tensor, thr_layout, thr_idx, projection);  // (M/2,K/1)

    std::cout << "Tensor: " << std::endl;
    cute::print(tensor);
    std::cout << std::endl;
    cute::print_tensor(tensor);
    std::cout << std::endl;
    cute::print_layout(tensor.layout());
    std::cout << std::endl;
    std::cout << "Thread Tensor: " << std::endl;
    cute::print(thrA);
    std::cout << std::endl;
    cute::print_tensor(thrA);
    std::cout << std::endl;
    cute::print_layout(thrA.layout());
    std::cout << std::endl;

    auto coord_1 = tile.get_flat_coord(0);
    auto coord_2 = tile.get_flat_coord(4);
    auto coord_3 = tile.get_flat_coord(16);
    auto coord_4 = tile.get_flat_coord(17);
    std::cout << "Flat coordinate 0: " << coord_1 << std::endl;
    std::cout << "Flat coordinate 4: " << coord_2 << std::endl;
    std::cout << "Flat coordinate 16: " << coord_3 << std::endl;
    std::cout << "Flat coordinate 17: " << coord_4 << std::endl;
}
