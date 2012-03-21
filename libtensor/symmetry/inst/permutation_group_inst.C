#include "../permutation_group.h"
#include "permutation_group_impl.h"

namespace libtensor {

template class permutation_group<1, double>;
template class permutation_group<2, double>;
template void permutation_group<2, double>::project_down(
        const mask<2> &msk, permutation_group<1, double> &);
template void permutation_group<2, double>::stabilize(
        const mask<2> (&msk)[1], permutation_group<2, double> &);
template void permutation_group<2, double>::stabilize(
        const mask<2> (&msk)[2], permutation_group<2, double> &);

template class permutation_group<3, double>;
template void permutation_group<3, double>::project_down(
        const mask<3> &msk, permutation_group<1, double> &);
template void permutation_group<3, double>::project_down(
        const mask<3> &msk, permutation_group<2, double> &);
template void permutation_group<3, double>::stabilize(
        const mask<3> (&msk)[1], permutation_group<3, double> &);
template void permutation_group<3, double>::stabilize(
        const mask<3> (&msk)[2], permutation_group<3, double> &);
template void permutation_group<3, double>::stabilize(
        const mask<3> (&msk)[3], permutation_group<3, double> &);

template class permutation_group<4, double>;
template void permutation_group<4, double>::project_down(
        const mask<4> &msk, permutation_group<1, double> &);
template void permutation_group<4, double>::project_down(
        const mask<4> &msk, permutation_group<2, double> &);
template void permutation_group<4, double>::project_down(
        const mask<4> &msk, permutation_group<3, double> &);
template void permutation_group<4, double>::stabilize(
        const mask<4> (&msk)[1], permutation_group<4, double> &);
template void permutation_group<4, double>::stabilize(
        const mask<4> (&msk)[2], permutation_group<4, double> &);
template void permutation_group<4, double>::stabilize(
        const mask<4> (&msk)[3], permutation_group<4, double> &);
template void permutation_group<4, double>::stabilize(
        const mask<4> (&msk)[4], permutation_group<4, double> &);

template class permutation_group<5, double>;
template void permutation_group<5, double>::project_down(
        const mask<5> &msk, permutation_group<1, double> &);
template void permutation_group<5, double>::project_down(
        const mask<5> &msk, permutation_group<2, double> &);
template void permutation_group<5, double>::project_down(
        const mask<5> &msk, permutation_group<3, double> &);
template void permutation_group<5, double>::project_down(
        const mask<5> &msk, permutation_group<4, double> &);
template void permutation_group<5, double>::stabilize(
        const mask<5> (&msk)[1], permutation_group<5, double> &);
template void permutation_group<5, double>::stabilize(
        const mask<5> (&msk)[2], permutation_group<5, double> &);
template void permutation_group<5, double>::stabilize(
        const mask<5> (&msk)[3], permutation_group<5, double> &);
template void permutation_group<5, double>::stabilize(
        const mask<5> (&msk)[4], permutation_group<5, double> &);
template void permutation_group<5, double>::stabilize(
        const mask<5> (&msk)[5], permutation_group<5, double> &);

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
template void permutation_group<6, double>::stabilize(
        const mask<6> (&msk)[1], permutation_group<6, double> &);
template void permutation_group<6, double>::stabilize(
        const mask<6> (&msk)[2], permutation_group<6, double> &);
template void permutation_group<6, double>::stabilize(
        const mask<6> (&msk)[3], permutation_group<6, double> &);
template void permutation_group<6, double>::stabilize(
        const mask<6> (&msk)[4], permutation_group<6, double> &);
template void permutation_group<6, double>::stabilize(
        const mask<6> (&msk)[5], permutation_group<6, double> &);
template void permutation_group<6, double>::stabilize(
        const mask<6> (&msk)[6], permutation_group<6, double> &);

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
template void permutation_group<7, double>::stabilize(
        const mask<7> (&msk)[1], permutation_group<7, double> &);
template void permutation_group<7, double>::stabilize(
        const mask<7> (&msk)[2], permutation_group<7, double> &);
template void permutation_group<7, double>::stabilize(
        const mask<7> (&msk)[3], permutation_group<7, double> &);
template void permutation_group<7, double>::stabilize(
        const mask<7> (&msk)[4], permutation_group<7, double> &);
template void permutation_group<7, double>::stabilize(
        const mask<7> (&msk)[5], permutation_group<7, double> &);
template void permutation_group<7, double>::stabilize(
        const mask<7> (&msk)[6], permutation_group<7, double> &);
template void permutation_group<7, double>::stabilize(
        const mask<7> (&msk)[7], permutation_group<7, double> &);

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
template void permutation_group<8, double>::stabilize(
        const mask<8> (&msk)[1], permutation_group<8, double> &);
template void permutation_group<8, double>::stabilize(
        const mask<8> (&msk)[2], permutation_group<8, double> &);
template void permutation_group<8, double>::stabilize(
        const mask<8> (&msk)[3], permutation_group<8, double> &);
template void permutation_group<8, double>::stabilize(
        const mask<8> (&msk)[4], permutation_group<8, double> &);
template void permutation_group<8, double>::stabilize(
        const mask<8> (&msk)[5], permutation_group<8, double> &);
template void permutation_group<8, double>::stabilize(
        const mask<8> (&msk)[6], permutation_group<8, double> &);
template void permutation_group<8, double>::stabilize(
        const mask<8> (&msk)[7], permutation_group<8, double> &);
template void permutation_group<8, double>::stabilize(
        const mask<8> (&msk)[8], permutation_group<8, double> &);

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
template void permutation_group<9, double>::stabilize(
        const mask<9> (&msk)[1], permutation_group<9, double> &);
template void permutation_group<9, double>::stabilize(
        const mask<9> (&msk)[2], permutation_group<9, double> &);
template void permutation_group<9, double>::stabilize(
        const mask<9> (&msk)[3], permutation_group<9, double> &);
template void permutation_group<9, double>::stabilize(
        const mask<9> (&msk)[4], permutation_group<9, double> &);
template void permutation_group<9, double>::stabilize(
        const mask<9> (&msk)[5], permutation_group<9, double> &);
template void permutation_group<9, double>::stabilize(
        const mask<9> (&msk)[6], permutation_group<9, double> &);
template void permutation_group<9, double>::stabilize(
        const mask<9> (&msk)[7], permutation_group<9, double> &);
template void permutation_group<9, double>::stabilize(
        const mask<9> (&msk)[8], permutation_group<9, double> &);
template void permutation_group<9, double>::stabilize(
        const mask<9> (&msk)[9], permutation_group<9, double> &);

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
template void permutation_group<10, double>::stabilize(
        const mask<10> (&msk)[1], permutation_group<10, double> &);
template void permutation_group<10, double>::stabilize(
        const mask<10> (&msk)[2], permutation_group<10, double> &);
template void permutation_group<10, double>::stabilize(
        const mask<10> (&msk)[3], permutation_group<10, double> &);
template void permutation_group<10, double>::stabilize(
        const mask<10> (&msk)[4], permutation_group<10, double> &);
template void permutation_group<10, double>::stabilize(
        const mask<10> (&msk)[5], permutation_group<10, double> &);
template void permutation_group<10, double>::stabilize(
        const mask<10> (&msk)[6], permutation_group<10, double> &);
template void permutation_group<10, double>::stabilize(
        const mask<10> (&msk)[7], permutation_group<10, double> &);
template void permutation_group<10, double>::stabilize(
        const mask<10> (&msk)[8], permutation_group<10, double> &);
template void permutation_group<10, double>::stabilize(
        const mask<10> (&msk)[9], permutation_group<10, double> &);
template void permutation_group<10, double>::stabilize(
        const mask<10> (&msk)[10], permutation_group<10, double> &);

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
template void permutation_group<11, double>::stabilize(
        const mask<11> (&msk)[1], permutation_group<11, double> &);
template void permutation_group<11, double>::stabilize(
        const mask<11> (&msk)[2], permutation_group<11, double> &);
template void permutation_group<11, double>::stabilize(
        const mask<11> (&msk)[3], permutation_group<11, double> &);
template void permutation_group<11, double>::stabilize(
        const mask<11> (&msk)[4], permutation_group<11, double> &);
template void permutation_group<11, double>::stabilize(
        const mask<11> (&msk)[5], permutation_group<11, double> &);
template void permutation_group<11, double>::stabilize(
        const mask<11> (&msk)[6], permutation_group<11, double> &);
template void permutation_group<11, double>::stabilize(
        const mask<11> (&msk)[7], permutation_group<11, double> &);
template void permutation_group<11, double>::stabilize(
        const mask<11> (&msk)[8], permutation_group<11, double> &);
template void permutation_group<11, double>::stabilize(
        const mask<11> (&msk)[9], permutation_group<11, double> &);
template void permutation_group<11, double>::stabilize(
        const mask<11> (&msk)[10], permutation_group<11, double> &);
template void permutation_group<11, double>::stabilize(
        const mask<11> (&msk)[11], permutation_group<11, double> &);

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
template void permutation_group<12, double>::stabilize(
        const mask<12> (&msk)[1], permutation_group<12, double> &);
template void permutation_group<12, double>::stabilize(
        const mask<12> (&msk)[2], permutation_group<12, double> &);
template void permutation_group<12, double>::stabilize(
        const mask<12> (&msk)[3], permutation_group<12, double> &);
template void permutation_group<12, double>::stabilize(
        const mask<12> (&msk)[4], permutation_group<12, double> &);
template void permutation_group<12, double>::stabilize(
        const mask<12> (&msk)[5], permutation_group<12, double> &);
template void permutation_group<12, double>::stabilize(
        const mask<12> (&msk)[6], permutation_group<12, double> &);
template void permutation_group<12, double>::stabilize(
        const mask<12> (&msk)[7], permutation_group<12, double> &);
template void permutation_group<12, double>::stabilize(
        const mask<12> (&msk)[8], permutation_group<12, double> &);
template void permutation_group<12, double>::stabilize(
        const mask<12> (&msk)[9], permutation_group<12, double> &);
template void permutation_group<12, double>::stabilize(
        const mask<12> (&msk)[10], permutation_group<12, double> &);
template void permutation_group<12, double>::stabilize(
        const mask<12> (&msk)[11], permutation_group<12, double> &);
template void permutation_group<12, double>::stabilize(
        const mask<12> (&msk)[12], permutation_group<12, double> &);

} // namespace libtensor

