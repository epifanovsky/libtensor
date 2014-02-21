#ifndef LIBTENSOR_SO_APPLY_SE_LABEL_IMPL_H
#define LIBTENSOR_SO_APPLY_SE_LABEL_IMPL_H

namespace libtensor {

template<size_t N, typename T>
const char *symmetry_operation_impl< so_apply<N, T>, se_label<N, T> >::k_clazz =
        "symmetry_operation_impl< so_apply<N, T>, se_label<N, T> >";

template<size_t N, typename T>
void symmetry_operation_impl< so_apply<N, T>, se_label<N, T> >::do_perform(
        symmetry_operation_params_t &params) const {

    typedef symmetry_element_set_adapter< N, T, se_label<N, T> > adapter_t;
    adapter_t g1(params.g1);
    params.g2.clear();

    for (typename adapter_t::iterator it1 = g1.begin();
            it1 != g1.end(); it1++) {

        se_label<N, T> e2(g1.get_elem(it1));
        e2.permute(params.perm1);

        if (! params.keep_zero) {
            evaluation_rule<N> r2;
            sequence<N, size_t> seq(1);
            product_rule<N> &pr2 = r2.new_product();
            pr2.add(seq, product_table_i::k_invalid);
            e2.set_rule(r2);
        }

        params.g2.insert(e2);
    }
}


} // namespace libtensor

#endif // LIBTENSOR_SO_APPLY_SE_LABEL_IMPL_H
