#include <memory> // for auto_ptr
#include <libtensor/kernels/kern_ddiv1.h>
#include <libtensor/kernels/kern_ddivadd1.h>
#include <libtensor/kernels/kern_dmul1.h>
#include <libtensor/kernels/kern_dmuladd1.h>
#include <libtensor/kernels/loop_list_runner.h>
#include <libtensor/diag_tensor/diag_tensor_ctrl.h>
#include "diag_tod_aux_constr_add.h"
#include "../diag_tod_mult1.h"

namespace libtensor {


template<size_t N>
const char *diag_tod_mult1<N>::k_clazz = "diag_tod_mult1<N>";


template<size_t N>
void diag_tod_mult1<N>::perform(bool zero, diag_tensor_wr_i<N, double> &dta) {

    diag_tod_mult1::start_timer();

    try {

        diag_tensor_wr_ctrl<N, double> ca(dta);
        diag_tensor_rd_ctrl<N, double> cb(m_dtb);

        //  For each subspace in A:
        //  1. Create a shadow subspace and place part of B constrained to it.
        //  2. Invoke multiplication/division kernel.
        //  3. Remove shadow subspace.

        const diag_tensor_space<N> &dtsa = dta.get_space();
        const diag_tensor_space<N> &dtsb = m_dtb.get_space();

        std::vector<size_t> ssla, sslb;
        dtsa.get_all_subspaces(ssla);
        dtsb.get_all_subspaces(sslb);

        for(size_t ssia = 0; ssia < ssla.size(); ssia++) {

            const diag_tensor_subspace<N> &ssa = dtsa.get_subspace(ssla[ssia]);

            //  Create & zero shadow subspace

            size_t ssna0 = ca.req_add_subspace(ssa);
            const diag_tensor_subspace<N> &ssa0 = dtsa.get_subspace(ssna0);
            size_t sza0 = dtsa.get_subspace_size(ssna0);
            double *pa0 = ca.req_dataptr(ssna0);
            for(size_t i = 0; i < sza0; i++) pa0[i] = 0.0;

            //  Copy part of B constrained to shadow subspace

            for(size_t ssib = 0; ssib < sslb.size(); ssib++) {

                const diag_tensor_subspace<N> &ssb =
                    dtsb.get_subspace(sslb[ssib]);

                    const double *pb = cb.req_const_dataptr(sslb[ssib]);
                    diag_tod_aux_constr_add<N>(dtsb.get_dims(), ssb, pb,
                        dtsb.get_subspace_size(sslb[ssib]), m_trb).
                        perform(ssa0, pa0, sza0);
                    cb.ret_const_dataptr(sslb[ssib], pb); pb = 0;
            }

            //  Invoke multiplication/division kernel

            double *pa = ca.req_dataptr(ssla[ssia]);

            std::list< loop_list_node<1, 1> > lpmul1, lpmul2;
            typename std::list< loop_list_node<1, 1> >::iterator imul =
                lpmul1.insert(lpmul1.end(), loop_list_node<1, 1>(sza0));
            imul->stepa(0) = 1;
            imul->stepb(0) = 1;

            loop_registers<1, 1> rmul;
            rmul.m_ptra[0] = pa0;
            rmul.m_ptrb[0] = pa;
            rmul.m_ptra_end[0] = pa0 + sza0;
            rmul.m_ptrb_end[0] = pa + sza0;

            std::auto_ptr< kernel_base<linalg, 1, 1> > kern_mul(
                m_recip ?
                    (zero ?
                        kern_ddiv1<linalg>::match(1.0, lpmul1, lpmul2) :
                        kern_ddivadd1::match(1.0, lpmul1, lpmul2)) :
                    (zero ?
                        kern_dmul1::match(1.0, lpmul1, lpmul2) :
                        kern_dmuladd1::match(1.0, lpmul1, lpmul2)));
            diag_tod_mult1::start_timer(kern_mul->get_name());
            loop_list_runner<linalg, 1, 1>(lpmul1).run(0, rmul, *kern_mul);
            diag_tod_mult1::stop_timer(kern_mul->get_name());

            ca.ret_dataptr(ssla[ssia], pa); pa = 0;
            ca.ret_dataptr(ssna0, pa0); pa0 = 0;

            //  Destroy shadow subspace

            ca.req_remove_subspace(ssna0);
        }

    } catch(...) {
        diag_tod_mult1::stop_timer();
        throw;
    }

    diag_tod_mult1::stop_timer();
}


} // namespace libtensor

