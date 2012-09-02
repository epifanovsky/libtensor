#ifndef LIBTENSOR_BTOD_SET_ELEM_H
#define LIBTENSOR_BTOD_SET_ELEM_H

#include <list>
#include <map>
#include "../defs.h"
#include "../core/abs_index.h"
#include "../core/orbit.h"
#include <libtensor/dense_tensor/tod_set.h>
#include <libtensor/dense_tensor/tod_set_elem.h>
#include <libtensor/block_tensor/block_tensor_i.h>
#include <libtensor/block_tensor/block_tensor_ctrl.h>

namespace libtensor {


/** \brief Sets a single element of a block %tensor to a value
    \tparam N Tensor order.

    The operation sets one block %tensor element specified by a block
    %index and an %index within the block. The symmetry is preserved.
    If the affected block shares an orbit with other blocks, those will
    be affected accordingly.

    Normally for clarity reasons the block %index used with this operation
    should be canonical. If it is not, the canonical block is changed using
    %symmetry rules such that the specified element of the specified block
    is given the specified value.

    \ingroup libtensor_btod
 **/
template<size_t N>
class btod_set_elem {
public:
    static const char *k_clazz; //!< Class name

private:
    typedef std::list< tensor_transf<N, double> > transf_list_t;
    typedef std::map<size_t, transf_list_t> transf_map_t;

public:
    /** \brief Default constructor
     **/
    btod_set_elem() { }

    /** \brief Performs the operation
        \param bt Block %tensor.
        \param bidx Block %index.
        \param idx Element %index within the block.
        \param d Element value.
     **/
    void perform(block_tensor_i<N, double> &bt, const index<N> &bidx,
        const index<N> &idx, double d);

private:
    bool make_transf_map(const symmetry<N, double> &sym,
        const dimensions<N> &bidims, const index<N> &idx,
        const tensor_transf<N, double> &tr, transf_map_t &alltransf);

private:
    btod_set_elem(const btod_set_elem<N> &);
    const btod_set_elem<N> &operator=(const btod_set_elem<N> &);

};


template<size_t N>
const char *btod_set_elem<N>::k_clazz = "btod_set_elem<N>";


template<size_t N>
void btod_set_elem<N>::perform(block_tensor_i<N, double> &bt,
    const index<N> &bidx, const index<N> &idx, double d) {

    static const char *method = "perform(block_tensor_i<N, double> &, "
        "const index<N> &, const index<N> &, double)";

    block_tensor_ctrl<N, double> ctrl(bt);

    dimensions<N> bidims(bt.get_bis().get_block_index_dims());
    orbit<N, double> o(ctrl.req_const_symmetry(), bidx);

    if (! o.is_allowed())
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
                "Block index not allowed by symmetry.");

    const tensor_transf<N, double> &tr = o.get_transf(bidx);
    abs_index<N> abidx(o.get_abs_canonical_index(), bidims);

    bool zero = ctrl.req_is_zero_block(abidx.get_index());
    dense_tensor_i<N, double> &blk = ctrl.req_block(abidx.get_index());

    if(zero) tod_set<N>().perform(blk);

    permutation<N> perm(tr.get_perm(), true);
    index<N> idx1(idx); idx1.permute(perm);
    double d1 = d / tr.get_scalar_tr().get_coeff();

    transf_map_t trmap;
    tensor_transf<N, double> tr0;
    make_transf_map(ctrl.req_const_symmetry(), bidims, abidx.get_index(),
        tr0, trmap);
    typename transf_map_t::iterator ilst =
        trmap.find(abidx.get_abs_index());
    for(typename transf_list_t::iterator itr = ilst->second.begin();
        itr != ilst->second.end(); itr++) {

        index<N> idx2(idx1);
        idx2.permute(itr->get_perm());
        double d2 = d1 * itr->get_scalar_tr().get_coeff();
        tod_set_elem<N>().perform(blk, idx2, d2);
    }

    ctrl.ret_block(abidx.get_index());
}


template<size_t N>
bool btod_set_elem<N>::make_transf_map(const symmetry<N, double> &sym,
    const dimensions<N> &bidims, const index<N> &idx,
    const tensor_transf<N, double> &tr, transf_map_t &alltransf) {

    size_t absidx = abs_index<N>::get_abs_index(idx, bidims);
    typename transf_map_t::iterator ilst = alltransf.find(absidx);
    if(ilst == alltransf.end()) {
        ilst = alltransf.insert(std::pair<size_t, transf_list_t>(
            absidx, transf_list_t())).first;
    }
    typename transf_list_t::iterator itr = ilst->second.begin();
    bool done = false;
    for(; itr != ilst->second.end(); itr++) {
        if(*itr == tr) {
            done = true;
            break;
        }
    }
    if(done) return true;
    ilst->second.push_back(tr);

    bool allowed = true;
    for(typename symmetry<N, double>::iterator iset = sym.begin();
        iset != sym.end(); iset++) {

        const symmetry_element_set<N, double> &eset =
            sym.get_subset(iset);
        for(typename symmetry_element_set<N, double>::const_iterator
            ielem = eset.begin(); ielem != eset.end(); ielem++) {

            const symmetry_element_i<N, double> &elem =
                eset.get_elem(ielem);
            index<N> idx2(idx);
            tensor_transf<N, double> tr2(tr);
            if(elem.is_allowed(idx2)) {
                elem.apply(idx2, tr2);
                allowed = make_transf_map(sym, bidims,
                    idx2, tr2, alltransf);
            } else {
                allowed = false;
            }
        }
    }
    return allowed;
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_SET_ELEM_H
