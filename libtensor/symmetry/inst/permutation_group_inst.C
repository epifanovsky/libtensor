#include "../permutation_group.h"
#include "permutation_group_impl.h"

namespace libtensor {

template class permutation_group<1, double>;
template class permutation_group<2, double>;
template void permutation_group<2, double>::project_down(
        const mask<2> &msk, permutation_group<1, double> &);

template class permutation_group<3, double>;
template void permutation_group<3, double>::project_down(
        const mask<3> &msk, permutation_group<1, double> &);
template void permutation_group<3, double>::project_down(
        const mask<3> &msk, permutation_group<2, double> &);

template class permutation_group<4, double>;
template void permutation_group<4, double>::project_down(
        const mask<4> &msk, permutation_group<1, double> &);
template void permutation_group<4, double>::project_down(
        const mask<4> &msk, permutation_group<2, double> &);
template void permutation_group<4, double>::project_down(
        const mask<4> &msk, permutation_group<3, double> &);

template class permutation_group<5, double>;
template void permutation_group<5, double>::project_down(
        const mask<5> &msk, permutation_group<1, double> &);
template void permutation_group<5, double>::project_down(
        const mask<5> &msk, permutation_group<2, double> &);
template void permutation_group<5, double>::project_down(
        const mask<5> &msk, permutation_group<3, double> &);
template void permutation_group<5, double>::project_down(
        const mask<5> &msk, permutation_group<4, double> &);

template class permutation_group<6, double>;
template void permutation_group<6, double>::project_down(
        const mask<6> &msk, permutation_group<1, double> &);
template void permutation_group<6, double>::project_down(
        const mask<6> &msk, permutation_group<2, double> &);
template void permutation_group<6, double>::project_down(
        const mask<6> &msk, permutation_group<3, double> &);
template void permutation_group<6, double>::project_down(
        const mask<6> &msk, permutation_group<4, double> &);
template void permutation_group<6, double>::project_down(
        const mask<6> &msk, permutation_group<5, double> &);

template class permutation_group<7, double>;
template void permutation_group<7, double>::project_down(
        const mask<7> &msk, permutation_group<1, double> &);
template void permutation_group<7, double>::project_down(
        const mask<7> &msk, permutation_group<2, double> &);
template void permutation_group<7, double>::project_down(
        const mask<7> &msk, permutation_group<3, double> &);
template void permutation_group<7, double>::project_down(
        const mask<7> &msk, permutation_group<4, double> &);
template void permutation_group<7, double>::project_down(
        const mask<7> &msk, permutation_group<5, double> &);
template void permutation_group<7, double>::project_down(
        const mask<7> &msk, permutation_group<6, double> &);

template class permutation_group<8, double>;
template void permutation_group<8, double>::project_down(
        const mask<8> &msk, permutation_group<1, double> &);
template void permutation_group<8, double>::project_down(
        const mask<8> &msk, permutation_group<2, double> &);
template void permutation_group<8, double>::project_down(
        const mask<8> &msk, permutation_group<3, double> &);
template void permutation_group<8, double>::project_down(
        const mask<8> &msk, permutation_group<4, double> &);
template void permutation_group<8, double>::project_down(
        const mask<8> &msk, permutation_group<5, double> &);
template void permutation_group<8, double>::project_down(
        const mask<8> &msk, permutation_group<6, double> &);
template void permutation_group<8, double>::project_down(
        const mask<8> &msk, permutation_group<7, double> &);

template class permutation_group<9, double>;
template void permutation_group<9, double>::project_down(
        const mask<9> &msk, permutation_group<1, double> &);
template void permutation_group<9, double>::project_down(
        const mask<9> &msk, permutation_group<2, double> &);
template void permutation_group<9, double>::project_down(
        const mask<9> &msk, permutation_group<3, double> &);
template void permutation_group<9, double>::project_down(
        const mask<9> &msk, permutation_group<4, double> &);
template void permutation_group<9, double>::project_down(
        const mask<9> &msk, permutation_group<5, double> &);
template void permutation_group<9, double>::project_down(
        const mask<9> &msk, permutation_group<6, double> &);
template void permutation_group<9, double>::project_down(
        const mask<9> &msk, permutation_group<7, double> &);
template void permutation_group<9, double>::project_down(
        const mask<9> &msk, permutation_group<8, double> &);

template class permutation_group<10, double>;
template void permutation_group<10, double>::project_down(
        const mask<10> &msk, permutation_group<1, double> &);
template void permutation_group<10, double>::project_down(
        const mask<10> &msk, permutation_group<2, double> &);
template void permutation_group<10, double>::project_down(
        const mask<10> &msk, permutation_group<3, double> &);
template void permutation_group<10, double>::project_down(
        const mask<10> &msk, permutation_group<4, double> &);
template void permutation_group<10, double>::project_down(
        const mask<10> &msk, permutation_group<5, double> &);
template void permutation_group<10, double>::project_down(
        const mask<10> &msk, permutation_group<6, double> &);
template void permutation_group<10, double>::project_down(
        const mask<10> &msk, permutation_group<7, double> &);
template void permutation_group<10, double>::project_down(
        const mask<10> &msk, permutation_group<8, double> &);
template void permutation_group<10, double>::project_down(
        const mask<10> &msk, permutation_group<9, double> &);

template class permutation_group<11, double>;
template void permutation_group<11, double>::project_down(
        const mask<11> &msk, permutation_group<1, double> &);
template void permutation_group<11, double>::project_down(
        const mask<11> &msk, permutation_group<2, double> &);
template void permutation_group<11, double>::project_down(
        const mask<11> &msk, permutation_group<3, double> &);
template void permutation_group<11, double>::project_down(
        const mask<11> &msk, permutation_group<4, double> &);
template void permutation_group<11, double>::project_down(
        const mask<11> &msk, permutation_group<5, double> &);
template void permutation_group<11, double>::project_down(
        const mask<11> &msk, permutation_group<6, double> &);
template void permutation_group<11, double>::project_down(
        const mask<11> &msk, permutation_group<7, double> &);
template void permutation_group<11, double>::project_down(
        const mask<11> &msk, permutation_group<8, double> &);
template void permutation_group<11, double>::project_down(
        const mask<11> &msk, permutation_group<9, double> &);
template void permutation_group<11, double>::project_down(
        const mask<11> &msk, permutation_group<10, double> &);

template class permutation_group<12, double>;
template void permutation_group<12, double>::project_down(
        const mask<12> &msk, permutation_group<1, double> &);
template void permutation_group<12, double>::project_down(
        const mask<12> &msk, permutation_group<2, double> &);
template void permutation_group<12, double>::project_down(
        const mask<12> &msk, permutation_group<3, double> &);
template void permutation_group<12, double>::project_down(
        const mask<12> &msk, permutation_group<4, double> &);
template void permutation_group<12, double>::project_down(
        const mask<12> &msk, permutation_group<5, double> &);
template void permutation_group<12, double>::project_down(
        const mask<12> &msk, permutation_group<6, double> &);
template void permutation_group<12, double>::project_down(
        const mask<12> &msk, permutation_group<7, double> &);
template void permutation_group<12, double>::project_down(
        const mask<12> &msk, permutation_group<8, double> &);
template void permutation_group<12, double>::project_down(
        const mask<12> &msk, permutation_group<9, double> &);
template void permutation_group<12, double>::project_down(
        const mask<12> &msk, permutation_group<10, double> &);
template void permutation_group<12, double>::project_down(
        const mask<12> &msk, permutation_group<11, double> &);

} // namespace libtensor

