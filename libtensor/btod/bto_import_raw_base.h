#ifndef LIBTENSOR_BTO_IMPORT_RAW_BASE_H
#define LIBTENSOR_BTO_IMPORT_RAW_BASE_H

#include <cmath> // for fabs
#include <sstream>
#include <libtensor/defs.h>
#include <libtensor/exception.h>
#include <libtensor/block_tensor/block_tensor_ctrl.h>
#include <libtensor/core/abs_index.h>
#include <libtensor/core/dimensions.h>
#include <libtensor/core/orbit.h>
#include <libtensor/core/orbit_list.h>
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/to_compare.h>
#include <libtensor/dense_tensor/to_copy.h>
#include <libtensor/symmetry/so_copy.h>
#include <libtensor/symmetry/bad_symmetry.h>

namespace libtensor {

/** \brief Base class for importing block tensors from external sources
    \tparam N Tensor order.
    \tparam Alloc Allocator for temporary buffers.

    This class provides utility methods for the routines that import block
    tensor data from an external source.

    verify_and_set_symmetry() verifies that the data placed in the block tensor
    satisfy the specified symmetry. Otherwise it raises bad_symmetry.

    check_zero() verifies that a tensor contains only zeros within the specified
    threshold.

    \ingroup libtensor_btod
 **/
template<size_t N, typename T, typename Alloc>
class bto_import_raw_base {
public:
    static const char *k_clazz; //!< Class name

public:
    /** \brief Default constructor
     **/
    bto_import_raw_base() { }

protected:
    /** \brief Verifies that the data in the given block tensor conforms with
            the symmetry; then installs that symmetry
     **/
    void verify_and_set_symmetry(block_tensor_i<N, T> &bt,
        const symmetry<N, T> &sym, T sym_thresh);

    /** \brief Checks that the given tensor only contains zeros within
            a threshold
     **/
    bool check_zero(dense_tensor_rd_i<N, T> &t, T thresh);

private:
    void verify_zero_orbit(block_tensor_rd_ctrl<N, T> &ctrl,
        const dimensions<N> &bidims, orbit<N, T> &o);
    void verify_nonzero_orbit(block_tensor_ctrl<N, T> &ctrl,
        const dimensions<N> &bidims, orbit<N, T> &o, T sym_thresh);

private:
    bto_import_raw_base(const bto_import_raw_base<N, T, Alloc>&);
    const bto_import_raw_base<N, T, Alloc> &operator=(
        const bto_import_raw_base<N, T, Alloc>&);

};


template<size_t N, typename T, typename Alloc>
const char *bto_import_raw_base<N, T, Alloc>::k_clazz =
    "bto_import_raw_base<N, T, Alloc>";


template<size_t N, typename T, typename Alloc>
void bto_import_raw_base<N, T, Alloc>::verify_and_set_symmetry(
    block_tensor_i<N, T> &bt, const symmetry<N, T> &sym,
    T sym_thresh) {

    static const char *method =
        "verify_and_set_symmetry(block_tensor_i<N, T>&, "
            "const symmetry<N, T>&, T)";

    block_tensor_ctrl<N, T> ctrl(bt);
    dimensions<N> bidims = bt.get_bis().get_block_index_dims();

    orbit_list<N, T> ol(sym);
    for(typename orbit_list<N, T>::iterator io = ol.begin();
        io != ol.end(); ++io) {

        orbit<N, T> o(sym, ol.get_abs_index(io));
        abs_index<N> aci(o.get_acindex(), bidims);

        if(ctrl.req_is_zero_block(aci.get_index())) {
            verify_zero_orbit(ctrl, bidims, o);
        } else {
            verify_nonzero_orbit(ctrl, bidims, o, sym_thresh);
        }
    }

    abs_index<N> ai(bidims);
    do {
        if (ol.contains(ai.get_abs_index())) continue;

        orbit<N, T> o(sym, ai.get_index());
	if (ai.get_abs_index() != o.get_acindex()) continue;

        if (! ctrl.req_is_zero_block(o.get_cindex())) {
            std::ostringstream ss;
            ss << "Non-zero block " << o.get_cindex() << ".";
            throw bad_symmetry(g_ns, k_clazz, method, __FILE__, __LINE__,
                ss.str().c_str());
        }
        verify_zero_orbit(ctrl, bidims, o);

    } while (ai.inc());

    so_copy<N, T>(sym).perform(ctrl.req_symmetry());
}


template<size_t N, typename T, typename Alloc>
void bto_import_raw_base<N, T, Alloc>::verify_zero_orbit(
    block_tensor_rd_ctrl<N, T> &ctrl, const dimensions<N> &bidims,
    orbit<N, T> &o) {

    static const char *method =
        "verify_zero_orbit(block_tensor_ctrl<N, T>&, "
            "const dimensions<N>&, orbit<N, T>&)";

    typedef typename orbit<N, T>::iterator iterator_t;

    for(iterator_t i = o.begin(); i != o.end(); ++i) {

        //  Skip the canonical block
        if(o.get_abs_index(i) == o.get_acindex()) continue;

        //  Make sure the block is strictly zero
        abs_index<N> ai(o.get_abs_index(i), bidims);
        if(!ctrl.req_is_zero_block(ai.get_index())) {
            abs_index<N> aci(o.get_acindex(), bidims);
            std::ostringstream ss;
            ss << "Asymmetry in zero block " << aci.get_index() << "->"
                << ai.get_index() << ".";
            throw bad_symmetry(g_ns, k_clazz, method, __FILE__, __LINE__,
                ss.str().c_str());
        }
    }
}


template<size_t N, typename T, typename Alloc>
void bto_import_raw_base<N, T, Alloc>::verify_nonzero_orbit(
    block_tensor_ctrl<N, T> &ctrl, const dimensions<N> &bidims,
    orbit<N, T> &o, T sym_thresh) {

    static const char *method =
        "verify_nonzero_orbit(block_tensor_ctrl<N, T>&, "
            "const dimensions<N>&, orbit<N, T>&, T)";

    typedef typename orbit<N, T>::iterator iterator_t;

    //  Get the canonical block
    abs_index<N> aci(o.get_acindex(), bidims);
    dense_tensor_rd_i<N, T> &cblk = ctrl.req_const_block(aci.get_index());

    for(iterator_t i = o.begin(); i != o.end(); ++i) {

        //  Skip the canonical block
        if(o.get_abs_index(i) == o.get_acindex()) continue;

        //  Current index and transformation
        abs_index<N> ai(o.get_abs_index(i), bidims);
        const tensor_transf<N, T> &tr = o.get_transf(i);

        //  Compare with the transformed canonical block
        dense_tensor_rd_i<N, T> &blk =
                ctrl.req_const_block(ai.get_index());
        dense_tensor<N, T, Alloc> tblk(blk.get_dims());
        to_copy<N, T> (cblk, tr.get_perm(), tr.get_scalar_tr().get_coeff()).
            perform(true, tblk);

        to_compare<N, T> cmp(blk, tblk, sym_thresh);
        if(!cmp.compare()) {

            ctrl.ret_const_block(ai.get_index());
            ctrl.ret_const_block(aci.get_index());

            std::ostringstream ss;
            ss << "Asymmetry in block " << aci.get_index() << "->"
                << ai.get_index() << " at element " << cmp.get_diff_index()
                << ": " << cmp.get_diff_elem_2() << " (expected), "
                << cmp.get_diff_elem_1() << " (found), "
                << cmp.get_diff_elem_1() - cmp.get_diff_elem_2() << " (diff).";
            throw bad_symmetry(g_ns, k_clazz, method, __FILE__, __LINE__,
                ss.str().c_str());
        }

        ctrl.ret_const_block(ai.get_index());

        //  Zero out the block with proper symmetry
        ctrl.req_zero_block(ai.get_index());
    }

    ctrl.ret_const_block(aci.get_index());
}


template<size_t N, typename T, typename Alloc>
bool bto_import_raw_base<N, T, Alloc>::check_zero(
        dense_tensor_rd_i<N, T> &t, T thresh) {

    dense_tensor_rd_ctrl<N, T> c(t);
    const T *p = c.req_const_dataptr();
    size_t sz = t.get_dims().get_size();
    bool ok = true;
    for(size_t i = 0; i < sz; i++) {
        if(fabs(p[i]) > thresh) {
            ok = false;
            break;
        }
    }
    c.ret_const_dataptr(p);
    return ok;
}


} // namespace libtensor

#endif // LIBTENSOR_BTO_IMPORT_RAW_BASE_H
