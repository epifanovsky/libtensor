#ifndef LIBTENSOR_BTOD_CONTRACT2_XM_IMPL_H
#define LIBTENSOR_BTOD_CONTRACT2_XM_IMPL_H

#include <stdexcept>
//#include <libmdc/threading_policy.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_add.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_copy.h>
#include <libtensor/gen_block_tensor/impl/gen_bto_contract2_impl.h>

#include <libtensor/core/impl/xm_allocator.h>
#include <libtensor/block_tensor/btod_contract2_xm.h>
#include <libtensor/libxm/xm.h>

namespace libtensor {


template<size_t N, size_t M, size_t K>
const char btod_contract2_xm_clazz<N, M, K>::k_clazz[] =
    "btod_contract2_xm<N, M, K>";


template<size_t N, size_t M, size_t K>
const char btod_contract2_xm<N, M, K>::k_clazz[] =
    "btod_contract2_xm<N, M, K>";


template<size_t N, size_t M, size_t K>
btod_contract2_xm<N, M, K>::btod_contract2_xm(
    const contraction2<N, M, K> &contr,
    block_tensor_rd_i<NA, double> &bta,
    block_tensor_rd_i<NB, double> &btb) :

    m_contr(contr), m_bta(bta), m_ka(), m_btb(btb), m_kb(), m_kc(),
    m_symc(contr, bta, btb),
    m_gbto(contr,
        bta, scalar_transf<double>(),
        btb, scalar_transf<double>(),
        scalar_transf<double>()) {
}


template<size_t N, size_t M, size_t K>
btod_contract2_xm<N, M, K>::btod_contract2_xm(
    const contraction2<N, M, K> &contr,
    block_tensor_rd_i<NA, double> &bta,
    double ka,
    block_tensor_rd_i<NB, double> &btb,
    double kb,
    double kc) :

    m_contr(contr), m_bta(bta), m_ka(ka), m_btb(btb), m_kb(kb), m_kc(kc),
    m_symc(contr, bta, btb),
    m_gbto(contr,
        bta, scalar_transf<double>(ka),
        btb, scalar_transf<double>(kb),
        scalar_transf<double>(kc)) {
}


template<size_t N>
xm_dim_t dimensions_to_xm(const dimensions<N> &dims)
{
    xm_dim_t d = xm_dim_zero(N);
    for (size_t i = 0; i < N; i++)
        d.i[i] = dims[N-i-1];
    return (d);
}


template<size_t NA, typename bti_traits>
dimensions<NA> get_tensor_dims(gen_block_tensor_rd_i<NA, bti_traits> &bta)
{
    block_index_space<NA> bisa(bta.get_bis());
    dimensions<NA> dims = bisa.get_block_index_dims();

    return (dims);
}


template<size_t NA, typename bti_traits>
dimensions<NA> get_tensor_part_dims(gen_block_tensor_rd_i<NA, bti_traits> &bta)
{
    block_index_space<NA> bisa(bta.get_bis());
    dimensions<NA> dims = bisa.get_block_index_dims();
    gen_block_tensor_rd_ctrl<NA, bti_traits> ctrl(bta);
    const symmetry<NA, double> &sym = ctrl.req_const_symmetry();

    for (typename symmetry<NA, double>::iterator i = sym.begin();
            i != sym.end(); i++) {
        if ((*i)->get_id() == "part") {
            if ((*i)->is_empty()) {
                throw std::runtime_error("part symmetry type is empty");
            }
            typename symmetry_element_set<NA, double>::const_iterator j =
                (*i)->begin();
            dimensions<NA> pdims =
                dynamic_cast<se_part<NA, double> *>(*j)->get_pdims();
            return (pdims);
        }
    }
    index<NA> i1, i2;
    return dimensions<NA>(index_range<NA>(i1, i2));
}


template<size_t NA, typename bti_traits>
void setup_input_tensor(struct xm_tensor *a, struct xm_allocator *allocator,
    gen_block_tensor_rd_i<NA, bti_traits> &bta) {

    gen_block_tensor_rd_ctrl<NA, bti_traits> ctrl(bta);
    block_index_space<NA> bisa(bta.get_bis());
    dimensions<NA> bisa_dims = bisa.get_block_index_dims();
    xm_dim_t dima = xm_tensor_get_dim(a);

    std::vector<size_t> nzblk;
    ctrl.req_nonzero_blocks(nzblk);

    for (size_t i = 0; i < nzblk.size(); i++) {
        abs_index<NA> lt_absidx(nzblk[i], bisa_dims);
        xm_dim_t idx = xm_dim_zero(NA);
        for (size_t j = 0; j < NA; j++)
            idx.i[NA-j-1] = lt_absidx.get_index()[j];
        dense_tensor_rd_i<NA, double> &lt_blk =
            ctrl.req_const_block(lt_absidx.get_index());
        {
            dimensions<NA> lt_blkdim =
                bisa.get_block_dims(lt_absidx.get_index());
            xm_dim_t blkdim = xm_dim_zero(NA);
            for (size_t j = 0; j < NA; j++)
                blkdim.i[j] = lt_blkdim[NA-j-1];
            uintptr_t data_ptr;
	    libtensor::allocator<double>::pointer_type p = dynamic_cast< dense_tensor<NA, double, libtensor::allocator<double> >& >(lt_blk).get_vm_ptr();
            memcpy(&data_ptr, &p, sizeof(data_ptr));
            xm_tensor_set_source_block(a, &idx, &blkdim, data_ptr);
        }
        ctrl.ret_const_block(lt_absidx.get_index());

        orbit<NA, double> o1(ctrl.req_const_symmetry(), nzblk[i]);
        for (typename orbit<NA, double>::iterator i1 = o1.begin();
                i1 != o1.end(); ++i1) {
            if (o1.get_abs_index(i1) == o1.get_acindex())
                continue;
            abs_index<NA> ai1(o1.get_abs_index(i1), bisa_dims);
            xm_dim_t idx2 = idx;
            for (size_t j = 0; j < NA; j++)
                idx2.i[NA-j-1] = ai1.get_index()[j];
            const tensor_transf<NA, double> &transf = o1.get_transf(i1);
            xm_dim_t perm = xm_dim_zero(NA);
            for (size_t j = 0; j < NA; j++) {
                size_t p = transf.get_perm()[j];
                perm.i[NA-j-1] = NA-p-1;
            }
            xm_tensor_set_block(a, &idx2, &idx, &perm,
                transf.get_scalar_tr().get_coeff());
        }
    }

    xm_dim_t dim = xm_tensor_get_dim(a);
    size_t nblk = xm_dim_dot(&dim);
    xm_dim_t idx = xm_dim_zero(NA);
    for (size_t i = 0; i < nblk; i++) {
        if (!xm_tensor_block_is_initialized(a, &idx)) {
            index<NA> lt_idx;
            for (size_t j = 0; j < NA; j++)
                lt_idx[j] = idx.i[NA-j-1];
            dimensions<NA> lt_blkdim = bisa.get_block_dims(lt_idx);
            xm_dim_t blkdim = xm_dim_zero(NA);
            for (size_t j = 0; j < NA; j++)
                blkdim.i[j] = lt_blkdim[NA-j-1];
            xm_tensor_set_zero_block(a, &idx, &blkdim);
        }
        xm_dim_inc(&idx, &dim);
    }
}


template<size_t NA, typename bti_traits>
void setup_output_tensor(struct xm_tensor *a, struct xm_allocator *allocator,
    gen_block_tensor_i<NA, bti_traits> &bta) {

    gen_block_tensor_ctrl<NA, bti_traits> ctrl(bta);
    block_index_space<NA> bisa(bta.get_bis());
    dimensions<NA> bisa_dims = bisa.get_block_index_dims();

    orbit_list<NA, double> ol1(ctrl.req_const_symmetry());
    for (typename orbit_list<NA, double>::iterator io1 = ol1.begin();
            io1 != ol1.end(); ++io1) {
        xm_dim_t idx = xm_dim_zero(NA);
        index<NA> lt_idx;
        ol1.get_index(io1, lt_idx);
        for (size_t j = 0; j < NA; j++)
            idx.i[NA-j-1] = lt_idx[j];
        dimensions<NA> lt_blkdim = bisa.get_block_dims(lt_idx);
        xm_dim_t blkdim = xm_dim_zero(NA);
        for (size_t j = 0; j < NA; j++)
            blkdim.i[j] = lt_blkdim[NA-j-1];

        dense_tensor_wr_i<NA, double> &lt_blk = ctrl.req_block(lt_idx);
        {
            tod_set<NA>().perform(true, lt_blk);
            uintptr_t data_ptr;
	    libtensor::allocator<double>::pointer_type p = dynamic_cast< dense_tensor<NA, double, libtensor::allocator<double> >& >(lt_blk).get_vm_ptr();
            memcpy(&data_ptr, &p, sizeof(data_ptr));
            xm_tensor_set_source_block(a, &idx, &blkdim, data_ptr);
        }
        ctrl.ret_block(lt_idx);

        orbit<NA, double> o1(ctrl.req_const_symmetry(),
            ol1.get_abs_index(io1));
        for (typename orbit<NA, double>::iterator i1 = o1.begin();
                i1 != o1.end(); ++i1) {
            if (o1.get_abs_index(i1) == o1.get_acindex())
                continue;
            abs_index<NA> ai1(o1.get_abs_index(i1), bisa_dims);
            xm_dim_t idx2 = idx;
            for (size_t j = 0; j < NA; j++)
                idx2.i[NA-j-1] = ai1.get_index()[j];
            const tensor_transf<NA, double> &transf = o1.get_transf(i1);
            xm_dim_t perm = xm_dim_zero(NA);
            for (size_t j = 0; j < NA; j++) {
                size_t p = transf.get_perm()[j];
                perm.i[NA-j-1] = NA-p-1;
            }
            xm_tensor_set_block(a, &idx2, &idx, &perm,
                transf.get_scalar_tr().get_coeff());
        }
    }

    xm_dim_t dim = xm_tensor_get_dim(a);
    size_t nblk = xm_dim_dot(&dim);
    xm_dim_t idx = xm_dim_zero(NA);
    for (size_t i = 0; i < nblk; i++) {
        if (!xm_tensor_block_is_initialized(a, &idx)) {
            index<NA> lt_idx;
            for (size_t j = 0; j < NA; j++)
                lt_idx[j] = idx.i[NA-j-1];
            dimensions<NA> lt_blkdim = bisa.get_block_dims(lt_idx);
            xm_dim_t blkdim = xm_dim_zero(NA);
            for (size_t j = 0; j < NA; j++)
                blkdim.i[j] = lt_blkdim[NA-j-1];
            xm_tensor_set_zero_block(a, &idx, &blkdim);
        }
        xm_dim_inc(&idx, &dim);
    }
}


template<size_t N, size_t M, size_t K>
void btod_contract2_xm<N, M, K>::perform(
    gen_block_stream_i<NC, bti_traits> &out) {

    block_tensor< NC, double, libtensor::allocator<double> >
        btc(m_symc.get_bis());

    {
        gen_block_tensor_ctrl<NC, bti_traits> ctrl(btc);
        so_copy<NC, double>(m_symc.get_symmetry()).perform(ctrl.req_symmetry());
    }

    perform(out, btc);
}


template<size_t N, size_t M, size_t K>
void btod_contract2_xm<N, M, K>::perform(
    gen_block_stream_i<NC, bti_traits> &out,
    gen_block_tensor_i<NC, bti_traits> &btc) {

    if (N + K > XM_MAX_DIM || M + K > XM_MAX_DIM || N + M > XM_MAX_DIM) {
        throw std::runtime_error("tensor n_dim > XM_MAX_DIM");
    }

    struct xm_allocator *allocator;
    struct xm_tensor *a, *b, *c;
    xm_dim_t dima, dimb, dimc, pdima, pdimb, pdimc;
    char idxbuf[32 * 3], *idxa, *idxb, *idxc;

    allocator = lt_xm_allocator::alloc_data::get_instance().xm_allocator_inst;

    idxa = idxbuf;
    idxb = idxa + 32;
    idxc = idxb + 32;

    dima = dimensions_to_xm(get_tensor_dims(m_bta));
    dimb = dimensions_to_xm(get_tensor_dims(m_btb));
    dimc = dimensions_to_xm(get_tensor_dims(btc));
    pdima = dimensions_to_xm(get_tensor_part_dims(m_bta));
    pdimb = dimensions_to_xm(get_tensor_part_dims(m_btb));
    pdimc = dimensions_to_xm(get_tensor_part_dims(btc));

    // setup contraction index strings
    char idx_letter = 'a';
    memset(idxbuf, 0, sizeof(idxbuf));
    for (size_t i = 0; i < NA; i++) {
        idxa[i] = idx_letter++;
    }
    for (size_t i = 0; i < NB; i++) {
        if (m_contr.get_conn()[NC+NA+i] < NC)
            idxb[i] = idx_letter++;
        else {
            size_t j = m_contr.get_conn()[NC+NA+i]-NC;
            idxb[i] = idxa[j];
        }
    }
    for (size_t i = 0; i < NC; i++) {
        if (m_contr.get_conn()[i] < NA+NC)
            idxc[i] = idxa[m_contr.get_conn()[i]-NC];
        else
            idxc[i] = idxb[m_contr.get_conn()[i]-NC-NA];
    }

    // libxm and libtensor use different storage order of data
    for (size_t i = 0; i < NA / 2; i++) {
        char t = idxa[i];
        idxa[i] = idxa[NA-i-1];
        idxa[NA-i-1] = t;
    }
    for (size_t i = 0; i < NB / 2; i++) {
        char t = idxb[i];
        idxb[i] = idxb[NB-i-1];
        idxb[NB-i-1] = t;
    }
    for (size_t i = 0; i < NC / 2; i++) {
        char t = idxc[i];
        idxc[i] = idxc[NC-i-1];
        idxc[NC-i-1] = t;
    }

    if ((a = xm_tensor_create(allocator, &dima, "a")) == NULL) {
        throw std::runtime_error("xm_tensor_create(a)");
    }
    if ((b = xm_tensor_create(allocator, &dimb, "b")) == NULL) {
        throw std::runtime_error("xm_tensor_create(b)");
    }
    if ((c = xm_tensor_create(allocator, &dimc, "c")) == NULL) {
        throw std::runtime_error("xm_tensor_create(c)");
    }

    xm_tensor_set_part_dim(a, &pdima);
    xm_tensor_set_part_dim(b, &pdimb);
    xm_tensor_set_part_dim(c, &pdimc);

    // setup tensor a
    setup_input_tensor<NA, bti_traits>(a, allocator, m_bta);
    // setup tensor b
    setup_input_tensor<NB, bti_traits>(b, allocator, m_btb);
    // setup tensor c
    setup_output_tensor<NC, bti_traits>(c, allocator, btc);

    //threading_policy::enable_blas_only();

    // contract
    double alpha = m_ka.get_coeff() * m_kb.get_coeff() * m_kc.get_coeff();
    if (xm_contract(alpha, a, b, 0.0, c, idxa, idxb, idxc)) {
        throw std::runtime_error("xm_contract");
    }

    //threading_policy::pop();

    {
        gen_block_tensor_ctrl<NC, bti_traits> ctrl(btc);
        orbit_list<NC, double> ol1(ctrl.req_const_symmetry());
        for (typename orbit_list<NC, double>::iterator io1 = ol1.begin();
                io1 != ol1.end(); ++io1) {
            index<NC> lt_idx;
            ol1.get_index(io1, lt_idx);
            {
                tensor_transf<NC, double> transf;
                dense_tensor_rd_i<NC, double> &lt_blk =
                    ctrl.req_const_block(lt_idx);
                out.put(lt_idx, lt_blk, transf);
                ctrl.ret_const_block(lt_idx);
            }
        }
    }

    // free all
    xm_tensor_free(a);
    xm_tensor_free(b);
    xm_tensor_free(c);
}


template<size_t N, size_t M, size_t K>
void btod_contract2_xm<N, M, K>::perform(
    gen_block_tensor_i<NC, bti_traits> &btc) {

    gen_bto_aux_copy<NC, btod_traits> out(get_symmetry(), btc);
    out.open();
    perform(out);
    out.close();
}


template<size_t N, size_t M, size_t K>
void btod_contract2_xm<N, M, K>::perform(
    gen_block_tensor_i<NC, bti_traits> &btc,
    const scalar_transf<double> &d) {

    typedef block_tensor_i_traits<double> bti_traits;

    gen_block_tensor_rd_ctrl<NC, bti_traits> cc(btc);
    std::vector<size_t> nzblkc;
    cc.req_nonzero_blocks(nzblkc);
    addition_schedule<NC, btod_traits> asch(get_symmetry(),
        cc.req_const_symmetry());
    asch.build(get_schedule(), nzblkc);

    gen_bto_aux_add<NC, btod_traits> out(get_symmetry(), asch, btc, d);
    out.open();
    perform(out);
    out.close();
}


template<size_t N, size_t M, size_t K>
void btod_contract2_xm<N, M, K>::perform(
    block_tensor_i<NC, double> &btc,
    double d) {

    perform(btc, scalar_transf<double>(d));
}


template<size_t N, size_t M, size_t K>
void btod_contract2_xm<N, M, K>::compute_block(
    bool zero,
    const index<NC> &ic,
    const tensor_transf<NC, double> &trc,
    dense_tensor_wr_i<NC, double> &blkc) {

    m_gbto.compute_block(zero, ic, trc, blkc);
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_CONTRACT2_XM_IMPL_H
