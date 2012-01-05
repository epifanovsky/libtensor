#ifndef LIBTENSOR_BTOD_IMPORT_RAW_BASE_H
#define LIBTENSOR_BTOD_IMPORT_RAW_BASE_H

#include <sstream>
#include "../defs.h"
#include "../exception.h"
#include "../core/abs_index.h"
#include "../core/dimensions.h"
#include "../core/block_tensor_ctrl.h"
#include "../core/orbit.h"
#include "../core/orbit_list.h"
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include "../core/tensor.h"
#include "../tod/tod_compare.h"
#include "../tod/tod_copy.h"
#include "../symmetry/so_copy.h"
#include "../symmetry/bad_symmetry.h"
#include "transf_double.h"

namespace libtensor {

/**	\brief Base class for importing block tensors from external sources
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
template<size_t N, typename Alloc>
class btod_import_raw_base {
public:
    static const char *k_clazz; //!< Class name

public:
    /** \brief Default constructor
     **/
    btod_import_raw_base() { }

protected:
    /** \brief Verifies that the data in the given block tensor conforms with
            the symmetry; then installs that symmetry
     **/
    void verify_and_set_symmetry(block_tensor_i<N, double> &bt,
        const symmetry<N, double> &sym, double sym_thresh);

    /** \brief Checks that the given tensor only contains zeros within
            a threshold
     **/
    bool check_zero(dense_tensor_i<N, double> &t, double thresh);

private:
    void verify_zero_orbit(block_tensor_ctrl<N, double> &ctrl,
        const dimensions<N> &bidims, orbit<N, double> &o);
    void verify_nonzero_orbit(block_tensor_ctrl<N, double> &ctrl,
        const dimensions<N> &bidims, orbit<N, double> &o, double sym_thresh);

private:
    btod_import_raw_base(const btod_import_raw_base<N, Alloc>&);
    const btod_import_raw_base<N, Alloc> &operator=(
        const btod_import_raw_base<N, Alloc>&);

};


template<size_t N, typename Alloc>
const char *btod_import_raw_base<N, Alloc>::k_clazz =
    "btod_import_raw_base<N, Alloc>";


template<size_t N, typename Alloc>
void btod_import_raw_base<N, Alloc>::verify_and_set_symmetry(
    block_tensor_i<N, double> &bt, const symmetry<N, double> &sym,
    double sym_thresh) {

    static const char *method =
        "verify_and_set_symmetry(block_tensor_i<N, double>&, "
            "const symmetry<N, double>&, double)";

    block_tensor_ctrl<N, double> ctrl(bt);
    dimensions<N> bidims = bt.get_bis().get_block_index_dims();

    orbit_list<N, double> ol(sym);
    for(typename orbit_list<N, double>::iterator io = ol.begin();
        io != ol.end(); ++io) {

        orbit<N, double> o(sym, ol.get_index(io));
        abs_index<N> aci(o.get_abs_canonical_index(), bidims);

        if(ctrl.req_is_zero_block(aci.get_index())) {
            verify_zero_orbit(ctrl, bidims, o);
        } else {
            verify_nonzero_orbit(ctrl, bidims, o, sym_thresh);
        }
    }

    so_copy<N, double>(sym).perform(ctrl.req_symmetry());
}


template<size_t N, typename Alloc>
void btod_import_raw_base<N, Alloc>::verify_zero_orbit(
    block_tensor_ctrl<N, double> &ctrl, const dimensions<N> &bidims,
    orbit<N, double> &o) {

    static const char *method =
        "verify_zero_orbit(block_tensor_ctrl<N, double>&, "
            "const dimensions<N>&, orbit<N, double>&)";

    typedef typename orbit<N, double>::iterator iterator_t;

    for(iterator_t i = o.begin(); i != o.end(); ++i) {

        //	Skip the canonical block
        if(o.get_abs_index(i) == o.get_abs_canonical_index()) continue;

        //	Make sure the block is strictly zero
        abs_index<N> ai(o.get_abs_index(i), bidims);
        if(!ctrl.req_is_zero_block(ai.get_index())) {
            abs_index<N> aci(o.get_abs_canonical_index(), bidims);
            std::ostringstream ss;
            ss << "Asymmetry in zero block " << aci.get_index() << "->"
                << ai.get_index() << ".";
            throw bad_symmetry(g_ns, k_clazz, method, __FILE__, __LINE__,
                ss.str().c_str());
        }
    }
}


template<size_t N, typename Alloc>
void btod_import_raw_base<N, Alloc>::verify_nonzero_orbit(
    block_tensor_ctrl<N, double> &ctrl, const dimensions<N> &bidims,
    orbit<N, double> &o, double sym_thresh) {

    static const char *method =
        "verify_nonzero_orbit(block_tensor_ctrl<N, double>&, "
            "const dimensions<N>&, orbit<N, double>&, double)";

    typedef typename orbit<N, double>::iterator iterator_t;

    cpu_pool cpus(1);

    //	Get the canonical block
    abs_index<N> aci(o.get_abs_canonical_index(), bidims);
    dense_tensor_i<N, double> &cblk = ctrl.req_block(aci.get_index());

    for(iterator_t i = o.begin(); i != o.end(); ++i) {

        //	Skip the canonical block
        if(o.get_abs_index(i) == o.get_abs_canonical_index()) continue;

        //	Current index and transformation
        abs_index<N> ai(o.get_abs_index(i), bidims);
        const transf<N, double> &tr = o.get_transf(i);

        //	Compare with the transformed canonical block
        dense_tensor_i<N, double> &blk = ctrl.req_block(ai.get_index());
        tensor<N, double, Alloc> tblk(blk.get_dims());
        tod_copy<N> (cblk, tr.get_perm(), tr.get_coeff()).
            perform(cpus, true, 1.0, tblk);

        tod_compare<N> cmp(blk, tblk, sym_thresh);
        if(!cmp.compare()) {

            ctrl.ret_block(ai.get_index());
            ctrl.ret_block(aci.get_index());

            std::ostringstream ss;
            ss << "Asymmetry in block " << aci.get_index() << "->"
                << ai.get_index() << " at element " << cmp.get_diff_index()
                << ": " << cmp.get_diff_elem_2() << " (expected), "
                << cmp.get_diff_elem_1() << " (found), "
                << cmp.get_diff_elem_1() - cmp.get_diff_elem_2() << " (diff).";
            throw bad_symmetry(g_ns, k_clazz, method, __FILE__, __LINE__,
                ss.str().c_str());
        }

        ctrl.ret_block(ai.get_index());

        //	Zero out the block with proper symmetry
        ctrl.req_zero_block(ai.get_index());
    }

    ctrl.ret_block(aci.get_index());
}


template<size_t N, typename Alloc>
bool btod_import_raw_base<N, Alloc>::check_zero(dense_tensor_i<N, double> &t,
    double thresh) {

    dense_tensor_ctrl<N, double> c(t);
    const double *p = c.req_const_dataptr();
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

#endif // LIBTENSOR_BTOD_IMPORT_RAW_BASE_H
