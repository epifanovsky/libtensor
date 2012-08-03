#include <libtensor/core/scalar_transf_double.h>
#include "../evaluation_rule.h"
#include "evaluation_rule_impl.h"

namespace libtensor {


template class evaluation_rule<1>;
template class evaluation_rule<2>;
template void evaluation_rule<2>::reduce(evaluation_rule<1> &res,
        const sequence<2, size_t> &rmap,
        const sequence<1, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<2>::merge(evaluation_rule<1> &res,
        const sequence<2, size_t> &mmap, const mask<1> &smsk) const;

template class evaluation_rule<3>;
template void evaluation_rule<3>::reduce(evaluation_rule<2> &res,
        const sequence<3, size_t> &rmap,
        const sequence<1, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<3>::reduce(evaluation_rule<1> &res,
        const sequence<3, size_t> &rmap,
        const sequence<2, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<3>::merge(evaluation_rule<1> &res,
        const sequence<3, size_t> &mmap, const mask<1> &smsk) const;
template void evaluation_rule<3>::merge(evaluation_rule<2> &res,
        const sequence<3, size_t> &mmap, const mask<2> &smsk) const;

template class evaluation_rule<4>;
template void evaluation_rule<4>::reduce(evaluation_rule<3> &res,
        const sequence<4, size_t> &rmap,
        const sequence<1, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<4>::reduce(evaluation_rule<2> &res,
        const sequence<4, size_t> &rmap,
        const sequence<2, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<4>::reduce(evaluation_rule<1> &res,
        const sequence<4, size_t> &rmap,
        const sequence<3, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<4>::merge(evaluation_rule<1> &res,
        const sequence<4, size_t> &mmap, const mask<1> &smsk) const;
template void evaluation_rule<4>::merge(evaluation_rule<2> &res,
        const sequence<4, size_t> &mmap, const mask<2> &smsk) const;
template void evaluation_rule<4>::merge(evaluation_rule<3> &res,
        const sequence<4, size_t> &mmap, const mask<3> &smsk) const;

template class evaluation_rule<5>;
template void evaluation_rule<5>::reduce(evaluation_rule<4> &res,
        const sequence<5, size_t> &rmap,
        const sequence<1, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<5>::reduce(evaluation_rule<3> &res,
        const sequence<5, size_t> &rmap,
        const sequence<2, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<5>::reduce(evaluation_rule<2> &res,
        const sequence<5, size_t> &rmap,
        const sequence<3, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<5>::reduce(evaluation_rule<1> &res,
        const sequence<5, size_t> &rmap,
        const sequence<4, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<5>::merge(evaluation_rule<1> &res,
        const sequence<5, size_t> &mmap, const mask<1> &smsk) const;
template void evaluation_rule<5>::merge(evaluation_rule<2> &res,
        const sequence<5, size_t> &mmap, const mask<2> &smsk) const;
template void evaluation_rule<5>::merge(evaluation_rule<3> &res,
        const sequence<5, size_t> &mmap, const mask<3> &smsk) const;
template void evaluation_rule<5>::merge(evaluation_rule<4> &res,
        const sequence<5, size_t> &mmap, const mask<4> &smsk) const;

template class evaluation_rule<6>;
template void evaluation_rule<6>::reduce(evaluation_rule<5> &res,
        const sequence<6, size_t> &rmap,
        const sequence<1, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<6>::reduce(evaluation_rule<4> &res,
        const sequence<6, size_t> &rmap,
        const sequence<2, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<6>::reduce(evaluation_rule<3> &res,
        const sequence<6, size_t> &rmap,
        const sequence<3, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<6>::reduce(evaluation_rule<2> &res,
        const sequence<6, size_t> &rmap,
        const sequence<4, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<6>::reduce(evaluation_rule<1> &res,
        const sequence<6, size_t> &rmap,
        const sequence<5, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<6>::merge(evaluation_rule<1> &res,
        const sequence<6, size_t> &mmap, const mask<1> &smsk) const;
template void evaluation_rule<6>::merge(evaluation_rule<2> &res,
        const sequence<6, size_t> &mmap, const mask<2> &smsk) const;
template void evaluation_rule<6>::merge(evaluation_rule<3> &res,
        const sequence<6, size_t> &mmap, const mask<3> &smsk) const;
template void evaluation_rule<6>::merge(evaluation_rule<4> &res,
        const sequence<6, size_t> &mmap, const mask<4> &smsk) const;
template void evaluation_rule<6>::merge(evaluation_rule<5> &res,
        const sequence<6, size_t> &mmap, const mask<5> &smsk) const;

template class evaluation_rule<7>;
template void evaluation_rule<7>::reduce(evaluation_rule<6> &res,
        const sequence<7, size_t> &rmap,
        const sequence<1, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<7>::reduce(evaluation_rule<5> &res,
        const sequence<7, size_t> &rmap,
        const sequence<2, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<7>::reduce(evaluation_rule<4> &res,
        const sequence<7, size_t> &rmap,
        const sequence<3, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<7>::reduce(evaluation_rule<3> &res,
        const sequence<7, size_t> &rmap,
        const sequence<4, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<7>::reduce(evaluation_rule<2> &res,
        const sequence<7, size_t> &rmap,
        const sequence<5, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<7>::reduce(evaluation_rule<1> &res,
        const sequence<7, size_t> &rmap,
        const sequence<6, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<7>::merge(evaluation_rule<1> &res,
        const sequence<7, size_t> &mmap, const mask<1> &smsk) const;
template void evaluation_rule<7>::merge(evaluation_rule<2> &res,
        const sequence<7, size_t> &mmap, const mask<2> &smsk) const;
template void evaluation_rule<7>::merge(evaluation_rule<3> &res,
        const sequence<7, size_t> &mmap, const mask<3> &smsk) const;
template void evaluation_rule<7>::merge(evaluation_rule<4> &res,
        const sequence<7, size_t> &mmap, const mask<4> &smsk) const;
template void evaluation_rule<7>::merge(evaluation_rule<5> &res,
        const sequence<7, size_t> &mmap, const mask<5> &smsk) const;
template void evaluation_rule<7>::merge(evaluation_rule<6> &res,
        const sequence<7, size_t> &mmap, const mask<6> &smsk) const;

template class evaluation_rule<8>;
template void evaluation_rule<8>::reduce(evaluation_rule<7> &res,
        const sequence<8, size_t> &rmap,
        const sequence<1, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<8>::reduce(evaluation_rule<6> &res,
        const sequence<8, size_t> &rmap,
        const sequence<2, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<8>::reduce(evaluation_rule<5> &res,
        const sequence<8, size_t> &rmap,
        const sequence<3, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<8>::reduce(evaluation_rule<4> &res,
        const sequence<8, size_t> &rmap,
        const sequence<4, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<8>::reduce(evaluation_rule<3> &res,
        const sequence<8, size_t> &rmap,
        const sequence<5, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<8>::reduce(evaluation_rule<2> &res,
        const sequence<8, size_t> &rmap,
        const sequence<6, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<8>::reduce(evaluation_rule<1> &res,
        const sequence<8, size_t> &rmap,
        const sequence<7, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<8>::merge(evaluation_rule<1> &res,
        const sequence<8, size_t> &mmap, const mask<1> &smsk) const;
template void evaluation_rule<8>::merge(evaluation_rule<2> &res,
        const sequence<8, size_t> &mmap, const mask<2> &smsk) const;
template void evaluation_rule<8>::merge(evaluation_rule<3> &res,
        const sequence<8, size_t> &mmap, const mask<3> &smsk) const;
template void evaluation_rule<8>::merge(evaluation_rule<4> &res,
        const sequence<8, size_t> &mmap, const mask<4> &smsk) const;
template void evaluation_rule<8>::merge(evaluation_rule<5> &res,
        const sequence<8, size_t> &mmap, const mask<5> &smsk) const;
template void evaluation_rule<8>::merge(evaluation_rule<6> &res,
        const sequence<8, size_t> &mmap, const mask<6> &smsk) const;
template void evaluation_rule<8>::merge(evaluation_rule<7> &res,
        const sequence<8, size_t> &mmap, const mask<7> &smsk) const;

template class evaluation_rule<9>;
template void evaluation_rule<9>::reduce(evaluation_rule<8> &res,
        const sequence<9, size_t> &rmap,
        const sequence<1, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<9>::reduce(evaluation_rule<7> &res,
        const sequence<9, size_t> &rmap,
        const sequence<2, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<9>::reduce(evaluation_rule<6> &res,
        const sequence<9, size_t> &rmap,
        const sequence<3, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<9>::reduce(evaluation_rule<5> &res,
        const sequence<9, size_t> &rmap,
        const sequence<4, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<9>::reduce(evaluation_rule<4> &res,
        const sequence<9, size_t> &rmap,
        const sequence<5, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<9>::reduce(evaluation_rule<3> &res,
        const sequence<9, size_t> &rmap,
        const sequence<6, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<9>::reduce(evaluation_rule<2> &res,
        const sequence<9, size_t> &rmap,
        const sequence<7, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<9>::reduce(evaluation_rule<1> &res,
        const sequence<9, size_t> &rmap,
        const sequence<8, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<9>::merge(evaluation_rule<1> &res,
        const sequence<9, size_t> &mmap, const mask<1> &smsk) const;
template void evaluation_rule<9>::merge(evaluation_rule<2> &res,
        const sequence<9, size_t> &mmap, const mask<2> &smsk) const;
template void evaluation_rule<9>::merge(evaluation_rule<3> &res,
        const sequence<9, size_t> &mmap, const mask<3> &smsk) const;
template void evaluation_rule<9>::merge(evaluation_rule<4> &res,
        const sequence<9, size_t> &mmap, const mask<4> &smsk) const;
template void evaluation_rule<9>::merge(evaluation_rule<5> &res,
        const sequence<9, size_t> &mmap, const mask<5> &smsk) const;
template void evaluation_rule<9>::merge(evaluation_rule<6> &res,
        const sequence<9, size_t> &mmap, const mask<6> &smsk) const;
template void evaluation_rule<9>::merge(evaluation_rule<7> &res,
        const sequence<9, size_t> &mmap, const mask<7> &smsk) const;
template void evaluation_rule<9>::merge(evaluation_rule<8> &res,
        const sequence<9, size_t> &mmap, const mask<8> &smsk) const;

template class evaluation_rule<10>;
template void evaluation_rule<10>::reduce(evaluation_rule<9> &res,
        const sequence<10, size_t> &rmap,
        const sequence<1, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<10>::reduce(evaluation_rule<8> &res,
        const sequence<10, size_t> &rmap,
        const sequence<2, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<10>::reduce(evaluation_rule<7> &res,
        const sequence<10, size_t> &rmap,
        const sequence<3, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<10>::reduce(evaluation_rule<6> &res,
        const sequence<10, size_t> &rmap,
        const sequence<4, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<10>::reduce(evaluation_rule<5> &res,
        const sequence<10, size_t> &rmap,
        const sequence<5, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<10>::reduce(evaluation_rule<4> &res,
        const sequence<10, size_t> &rmap,
        const sequence<6, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<10>::reduce(evaluation_rule<3> &res,
        const sequence<10, size_t> &rmap,
        const sequence<7, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<10>::reduce(evaluation_rule<2> &res,
        const sequence<10, size_t> &rmap,
        const sequence<8, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<10>::reduce(evaluation_rule<1> &res,
        const sequence<10, size_t> &rmap,
        const sequence<9, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<10>::merge(evaluation_rule<1> &res,
        const sequence<10, size_t> &mmap, const mask<1> &smsk) const;
template void evaluation_rule<10>::merge(evaluation_rule<2> &res,
        const sequence<10, size_t> &mmap, const mask<2> &smsk) const;
template void evaluation_rule<10>::merge(evaluation_rule<3> &res,
        const sequence<10, size_t> &mmap, const mask<3> &smsk) const;
template void evaluation_rule<10>::merge(evaluation_rule<4> &res,
        const sequence<10, size_t> &mmap, const mask<4> &smsk) const;
template void evaluation_rule<10>::merge(evaluation_rule<5> &res,
        const sequence<10, size_t> &mmap, const mask<5> &smsk) const;
template void evaluation_rule<10>::merge(evaluation_rule<6> &res,
        const sequence<10, size_t> &mmap, const mask<6> &smsk) const;
template void evaluation_rule<10>::merge(evaluation_rule<7> &res,
        const sequence<10, size_t> &mmap, const mask<7> &smsk) const;
template void evaluation_rule<10>::merge(evaluation_rule<8> &res,
        const sequence<10, size_t> &mmap, const mask<8> &smsk) const;
template void evaluation_rule<10>::merge(evaluation_rule<9> &res,
        const sequence<10, size_t> &mmap, const mask<9> &smsk) const;

template class evaluation_rule<11>;
template void evaluation_rule<11>::reduce(evaluation_rule<10> &res,
        const sequence<11, size_t> &rmap,
        const sequence<1, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<11>::reduce(evaluation_rule<9> &res,
        const sequence<11, size_t> &rmap,
        const sequence<2, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<11>::reduce(evaluation_rule<8> &res,
        const sequence<11, size_t> &rmap,
        const sequence<3, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<11>::reduce(evaluation_rule<7> &res,
        const sequence<11, size_t> &rmap,
        const sequence<4, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<11>::reduce(evaluation_rule<6> &res,
        const sequence<11, size_t> &rmap,
        const sequence<5, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<11>::reduce(evaluation_rule<5> &res,
        const sequence<11, size_t> &rmap,
        const sequence<6, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<11>::reduce(evaluation_rule<4> &res,
        const sequence<11, size_t> &rmap,
        const sequence<7, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<11>::reduce(evaluation_rule<3> &res,
        const sequence<11, size_t> &rmap,
        const sequence<8, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<11>::reduce(evaluation_rule<2> &res,
        const sequence<11, size_t> &rmap,
        const sequence<9, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<11>::reduce(evaluation_rule<1> &res,
        const sequence<11, size_t> &rmap,
        const sequence<10, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<11>::merge(evaluation_rule<1> &res,
        const sequence<11, size_t> &mmap, const mask<1> &smsk) const;
template void evaluation_rule<11>::merge(evaluation_rule<2> &res,
        const sequence<11, size_t> &mmap, const mask<2> &smsk) const;
template void evaluation_rule<11>::merge(evaluation_rule<3> &res,
        const sequence<11, size_t> &mmap, const mask<3> &smsk) const;
template void evaluation_rule<11>::merge(evaluation_rule<4> &res,
        const sequence<11, size_t> &mmap, const mask<4> &smsk) const;
template void evaluation_rule<11>::merge(evaluation_rule<5> &res,
        const sequence<11, size_t> &mmap, const mask<5> &smsk) const;
template void evaluation_rule<11>::merge(evaluation_rule<6> &res,
        const sequence<11, size_t> &mmap, const mask<6> &smsk) const;
template void evaluation_rule<11>::merge(evaluation_rule<7> &res,
        const sequence<11, size_t> &mmap, const mask<7> &smsk) const;
template void evaluation_rule<11>::merge(evaluation_rule<8> &res,
        const sequence<11, size_t> &mmap, const mask<8> &smsk) const;
template void evaluation_rule<11>::merge(evaluation_rule<9> &res,
        const sequence<11, size_t> &mmap, const mask<9> &smsk) const;
template void evaluation_rule<11>::merge(evaluation_rule<10> &res,
        const sequence<11, size_t> &mmap, const mask<10> &smsk) const;

template class evaluation_rule<12>;
template void evaluation_rule<12>::reduce(evaluation_rule<11> &res,
        const sequence<12, size_t> &rmap,
        const sequence<1, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<12>::reduce(evaluation_rule<10> &res,
        const sequence<12, size_t> &rmap,
        const sequence<2, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<12>::reduce(evaluation_rule<9> &res,
        const sequence<12, size_t> &rmap,
        const sequence<3, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<12>::reduce(evaluation_rule<8> &res,
        const sequence<12, size_t> &rmap,
        const sequence<4, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<12>::reduce(evaluation_rule<7> &res,
        const sequence<12, size_t> &rmap,
        const sequence<5, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<12>::reduce(evaluation_rule<6> &res,
        const sequence<12, size_t> &rmap,
        const sequence<6, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<12>::reduce(evaluation_rule<5> &res,
        const sequence<12, size_t> &rmap,
        const sequence<7, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<12>::reduce(evaluation_rule<4> &res,
        const sequence<12, size_t> &rmap,
        const sequence<8, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<12>::reduce(evaluation_rule<3> &res,
        const sequence<12, size_t> &rmap,
        const sequence<9, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<12>::reduce(evaluation_rule<2> &res,
        const sequence<12, size_t> &rmap,
        const sequence<10, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<12>::reduce(evaluation_rule<1> &res,
        const sequence<12, size_t> &rmap,
        const sequence<11, label_group_t> &rdims,
        const product_table_i &pt) const;
template void evaluation_rule<12>::merge(evaluation_rule<1> &res,
        const sequence<12, size_t> &mmap, const mask<1> &smsk) const;
template void evaluation_rule<12>::merge(evaluation_rule<2> &res,
        const sequence<12, size_t> &mmap, const mask<2> &smsk) const;
template void evaluation_rule<12>::merge(evaluation_rule<3> &res,
        const sequence<12, size_t> &mmap, const mask<3> &smsk) const;
template void evaluation_rule<12>::merge(evaluation_rule<4> &res,
        const sequence<12, size_t> &mmap, const mask<4> &smsk) const;
template void evaluation_rule<12>::merge(evaluation_rule<5> &res,
        const sequence<12, size_t> &mmap, const mask<5> &smsk) const;
template void evaluation_rule<12>::merge(evaluation_rule<6> &res,
        const sequence<12, size_t> &mmap, const mask<6> &smsk) const;
template void evaluation_rule<12>::merge(evaluation_rule<7> &res,
        const sequence<12, size_t> &mmap, const mask<7> &smsk) const;
template void evaluation_rule<12>::merge(evaluation_rule<8> &res,
        const sequence<12, size_t> &mmap, const mask<8> &smsk) const;
template void evaluation_rule<12>::merge(evaluation_rule<9> &res,
        const sequence<12, size_t> &mmap, const mask<9> &smsk) const;
template void evaluation_rule<12>::merge(evaluation_rule<10> &res,
        const sequence<12, size_t> &mmap, const mask<10> &smsk) const;
template void evaluation_rule<12>::merge(evaluation_rule<11> &res,
        const sequence<12, size_t> &mmap, const mask<11> &smsk) const;

template class evaluation_rule<13>;
template class evaluation_rule<14>;
template class evaluation_rule<15>;
template class evaluation_rule<16>;


} // namespace libtensor

