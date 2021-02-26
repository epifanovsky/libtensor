#ifndef LIBTENSOR_BTO_CONTRACT2_XM_IMPL_H
#define LIBTENSOR_BTO_CONTRACT2_XM_IMPL_H

#include <stdexcept>
#include <libmdc/threading_policy.h>
//#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/core/scalar_transf.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_add.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_copy.h>
#include <libtensor/gen_block_tensor/impl/gen_bto_contract2_impl.h>

#include <libtensor/core/impl/xm_allocator.h>
#include <libtensor/block_tensor/bto_contract2_xm.h>
#include <libtensor/libxm/src/xm.h>
#include <libtensor/libxm/src/scalar.h>

namespace libtensor {

template<size_t N, size_t M, size_t K, typename T>
const char bto_contract2_xm_clazz<N, M, K, T>::k_clazz[] =
    "bto_contract2_xm<N, M, K, T>";

template<size_t N, size_t M, size_t K, typename T>
const char bto_contract2_xm<N, M, K, T>::k_clazz[] =
    "bto_contract2_xm<N, M, K, T>";

template<size_t N, size_t M, size_t K, typename T>
bto_contract2_xm<N, M, K, T>::bto_contract2_xm(
    const contraction2<N, M, K> &contr,
    block_tensor_rd_i<NA, T> &bta,
    block_tensor_rd_i<NB, T> &btb) :

    m_contr(contr), m_bta(bta), m_ka(), m_btb(btb), m_kb(), m_kc(),
    m_symc(contr, bta, btb),
    m_gbto(contr,
        bta, scalar_transf<T>(),
        btb, scalar_transf<T>(),
        scalar_transf<T>()) {
}

template<size_t N, size_t M, size_t K, typename T>
bto_contract2_xm<N, M, K, T>::bto_contract2_xm(
    const contraction2<N, M, K> &contr,
    block_tensor_rd_i<NA, T> &bta,
    T ka,
    block_tensor_rd_i<NB, T> &btb,
    T kb,
    T kc) :

    m_contr(contr), m_bta(bta), m_ka(ka), m_btb(btb), m_kb(kb), m_kc(kc),
    m_symc(contr, bta, btb),
    m_gbto(contr,
        bta, scalar_transf<T>(ka),
        btb, scalar_transf<T>(kb),
        scalar_transf<T>(kc)) {
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
xm_block_space_t *make_blockspace(gen_block_tensor_rd_i<NA, bti_traits> &bta)
{
    block_index_space<NA> bisa(bta.get_bis());
    dimensions<NA> ltdims = bisa.get_dims();
    xm_dim_t dims = dimensions_to_xm(ltdims);
    xm_block_space_t *bs = xm_block_space_create(dims);
    for (size_t i = 0; i < NA; i++) {
        split_points sp = bisa.get_splits(bisa.get_type(i));
        for (size_t j = 0; j < sp.get_num_points(); j++) {
            xm_block_space_split(bs, NA-i-1, sp[j]);
	}
    }
    return bs;
}

/*
template<size_t NA, typename T, typename bti_traits>
void setup_input_tensor(struct xm_tensor *a, struct xm_allocator *allocator,
    gen_block_tensor_rd_i<NA, bti_traits> &bta) {}
*/

template<size_t NA, typename T, typename bti_traits>
void setup_input_tensor(struct xm_tensor *a, struct xm_allocator *allocator,
    gen_block_tensor_rd_i<NA, bti_traits> &bta) {

    gen_block_tensor_rd_ctrl<NA, bti_traits> ctrl(bta);
    block_index_space<NA> bisa(bta.get_bis());
    dimensions<NA> bisa_dims = bisa.get_block_index_dims();

    std::vector<size_t> nzblk;
    ctrl.req_nonzero_blocks(nzblk);

    for (size_t i = 0; i < nzblk.size(); i++) {
        abs_index<NA> lt_absidx(nzblk[i], bisa_dims);
        xm_dim_t idx = xm_dim_zero(NA);
        for (size_t j = 0; j < NA; j++)
            idx.i[NA-j-1] = lt_absidx.get_index()[j];
        //dense_tensor_rd_i<NA, double> &lt_blk =
        dense_tensor_rd_i<NA, T> &lt_blk =
            ctrl.req_const_block(lt_absidx.get_index());
        {
            dimensions<NA> lt_blkdim =
                bisa.get_block_dims(lt_absidx.get_index());
            xm_dim_t blkdim = xm_dim_zero(NA);
            for (size_t j = 0; j < NA; j++)
                blkdim.i[j] = lt_blkdim[NA-j-1];
            uintptr_t data_ptr;
            //libtensor::allocator::pointer_type p = dynamic_cast< dense_tensor<NA, double, libtensor::allocator >& >(lt_blk).get_vm_ptr();
            libtensor::allocator::pointer_type p = dynamic_cast< dense_tensor<NA, T, libtensor::allocator >& >(lt_blk).get_vm_ptr();
            memcpy(&data_ptr, &p, sizeof(data_ptr));
            xm_tensor_set_canonical_block_raw(a, idx, data_ptr);
        }
        ctrl.ret_const_block(lt_absidx.get_index());

        //orbit<NA, double> o1(ctrl.req_const_symmetry(), nzblk[i]);
        orbit<NA, T> o1(ctrl.req_const_symmetry(), nzblk[i]);
        //for (typename orbit<NA, double>::iterator i1 = o1.begin();
        for (typename orbit<NA, T>::iterator i1 = o1.begin();
                i1 != o1.end(); ++i1) {
            if (o1.get_abs_index(i1) == o1.get_acindex())
                continue;
            abs_index<NA> ai1(o1.get_abs_index(i1), bisa_dims);
            xm_dim_t idx2 = idx;
            for (size_t j = 0; j < NA; j++)
                idx2.i[NA-j-1] = ai1.get_index()[j];
            //const tensor_transf<NA, double> &transf = o1.get_transf(i1);
            const tensor_transf<NA, T> &transf = o1.get_transf(i1);
            xm_dim_t perm = xm_dim_zero(NA);
            for (size_t j = 0; j < NA; j++) {
                size_t p = transf.get_perm()[j];
                perm.i[NA-j-1] = NA-p-1;
            }
            xm_tensor_set_derivative_block(a, idx2, idx, perm,
                transf.get_scalar_tr().get_coeff());
        }
    }
}

template<size_t NC, typename T, typename bti_traits>
void setup_output_tensor(struct xm_tensor *a, struct xm_allocator *allocator,
    gen_block_tensor_i<NC, bti_traits> &bta,
    const assignment_schedule<NC, T> &sch) {

    gen_block_tensor_ctrl<NC, bti_traits> ctrl(bta);
    block_index_space<NC> bisa(bta.get_bis());
    dimensions<NC> bisa_dims = bisa.get_block_index_dims();

    for (typename assignment_schedule<NC, T>::iterator i =
        sch.begin(); i != sch.end(); ++i) {

        xm_dim_t idx = xm_dim_zero(NC);
        index<NC> lt_idx;
        abs_index<NC>::get_index(sch.get_abs_index(i), bisa_dims, lt_idx);
        for (size_t j = 0; j < NC; j++)
            idx.i[NC-j-1] = lt_idx[j];
        dimensions<NC> lt_blkdim = bisa.get_block_dims(lt_idx);
        xm_dim_t blkdim = xm_dim_zero(NC);
        for (size_t j = 0; j < NC; j++)
            blkdim.i[j] = lt_blkdim[NC-j-1];

        dense_tensor_wr_i<NC, T> &lt_blk = ctrl.req_block(lt_idx);
        {
            uintptr_t data_ptr;
            libtensor::allocator::pointer_type p = dynamic_cast< dense_tensor<NC, T, libtensor::allocator >& >(lt_blk).get_vm_ptr();
            memcpy(&data_ptr, &p, sizeof(data_ptr));
            xm_tensor_set_canonical_block_raw(a, idx, data_ptr);
        }
        ctrl.ret_block(lt_idx);
    }
}

template<size_t NA, typename T, typename bti_traits>
void setup_output_tensor_partial(struct xm_tensor *a,
    struct xm_allocator *allocator,
    gen_block_tensor_i<NA, bti_traits> &bta,
    const std::vector<size_t> &blklist) {

    gen_block_tensor_ctrl<NA, bti_traits> ctrl(bta);
    block_index_space<NA> bisa(bta.get_bis());
    dimensions<NA> bisa_dims = bisa.get_block_index_dims();

    for (size_t i = 0; i < blklist.size(); i++) {
        abs_index<NA> ai1(blklist[i], bisa_dims);
        xm_dim_t idx = xm_dim_zero(NA);
        for (size_t j = 0; j < NA; j++)
            idx.i[NA-j-1] = ai1.get_index()[j];
        dense_tensor_wr_i<NA, T> &lt_blk = ctrl.req_block(ai1.get_index());
        {
            uintptr_t data_ptr;
            libtensor::allocator::pointer_type p = dynamic_cast< dense_tensor<NA, T, libtensor::allocator >& >(lt_blk).get_vm_ptr();
            memcpy(&data_ptr, &p, sizeof(data_ptr));
            xm_tensor_set_canonical_block_raw(a, idx, data_ptr);
        }
        ctrl.ret_block(ai1.get_index());
    }
}

template<size_t N, size_t M, size_t K, typename T>
void bto_contract2_xm<N, M, K, T>::perform(
    gen_block_stream_i<NC, bti_traits> &out) {

    block_tensor< NC, T, libtensor::allocator >
        btc(m_symc.get_bis());

    perform(btc);

    {
        gen_block_tensor_ctrl<NC, bti_traits> ctrl(btc);
        dimensions<NC> bidimsc = m_symc.get_bis().get_block_index_dims();
        const assignment_schedule<NC, T> &sch = get_schedule();
        for (typename assignment_schedule<NC, T>::iterator i =
            sch.begin(); i != sch.end(); ++i) {
            index<NC> lt_idx;
            abs_index<NC>::get_index(sch.get_abs_index(i), bidimsc, lt_idx);
            {
                tensor_transf<NC, T> transf;
                dense_tensor_rd_i<NC, T> &lt_blk =
                    ctrl.req_const_block(lt_idx);
                out.put(lt_idx, lt_blk, transf);
                ctrl.ret_const_block(lt_idx);
            }
        }
    }
}

template<size_t N, size_t M, size_t K, typename T>
void bto_contract2_xm<N, M, K, T>::perform(
    gen_block_tensor_i<NC, bti_traits> &btc) {

    if (N + K > XM_MAX_DIM || M + K > XM_MAX_DIM || N + M > XM_MAX_DIM) {
        throw std::runtime_error("tensor n_dim > XM_MAX_DIM");
    }

    xm_allocator_t *allocator;
    xm_block_space_t *bsa, *bsb, *bsc;
    xm_tensor_t *a, *b, *c;
    char idxbuf[32 * 3], *idxa, *idxb, *idxc;
    xm_scalar_type_t type = (typeid(T) == typeid(double)) ? XM_SCALAR_DOUBLE : XM_SCALAR_FLOAT;

    allocator = lt_xm_allocator::alloc_data::get_instance().xm_allocator_inst;

    idxa = idxbuf;
    idxb = idxa + 32;
    idxc = idxb + 32;

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

    bsa = make_blockspace(m_bta);
    bsb = make_blockspace(m_btb);
    bsc = make_blockspace(btc);

    if ((a = xm_tensor_create(bsa, type, allocator)) == NULL) {
        throw std::runtime_error("xm_tensor_create");
    }
    if ((b = xm_tensor_create(bsb, type, allocator)) == NULL) {
        throw std::runtime_error("xm_tensor_create");
    }
    if ((c = xm_tensor_create(bsc, type, allocator)) == NULL) {
        throw std::runtime_error("xm_tensor_create");
    }
    xm_block_space_free(bsa);
    xm_block_space_free(bsb);
    xm_block_space_free(bsc);

    // setup tensor a
    setup_input_tensor<NA, T, bti_traits>(a, allocator, m_bta);
    // setup tensor b
    setup_input_tensor<NB, T, bti_traits>(b, allocator, m_btb);
    // setup tensor c
    setup_output_tensor<NC, T, bti_traits>(c, allocator, btc, get_schedule());

    threading_policy::enable_omp_only();

    // contract
    double alpha = m_ka.get_coeff() * m_kb.get_coeff() * m_kc.get_coeff();
    xm_contract(alpha, a, b, 0.0, c, idxa, idxb, idxc);

    threading_policy::pop();

    // free all
    xm_tensor_free(a);
    xm_tensor_free(b);
    xm_tensor_free(c);
}

template<size_t N, size_t M, size_t K, typename T>
void bto_contract2_xm<N, M, K, T>::partial(
     gen_block_tensor_i<NC, bti_traits> &btc,
    const std::vector<size_t> &blklist) {
 
    if (N + K > XM_MAX_DIM || M + K > XM_MAX_DIM || N + M > XM_MAX_DIM) {
        throw std::runtime_error("tensor n_dim > XM_MAX_DIM");
    }
 
    xm_allocator_t *allocator;
    xm_block_space_t *bsa, *bsb, *bsc;
    xm_tensor_t *a, *b, *c;
    char idxbuf[32 * 3], *idxa, *idxb, *idxc;
    xm_scalar_type_t type = (typeid(T) == typeid(double)) ? XM_SCALAR_DOUBLE : XM_SCALAR_FLOAT;

    allocator = lt_xm_allocator::alloc_data::get_instance().xm_allocator_inst;

    idxa = idxbuf;
    idxb = idxa + 32;
    idxc = idxb + 32;

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

    bsa = make_blockspace(m_bta);
    bsb = make_blockspace(m_btb);
    bsc = make_blockspace(btc);

    if ((a = xm_tensor_create(bsa, type, allocator)) == NULL) {
        throw std::runtime_error("xm_tensor_create");
    }
    if ((b = xm_tensor_create(bsb, type, allocator)) == NULL) {
        throw std::runtime_error("xm_tensor_create");
    }
    if ((c = xm_tensor_create(bsc, type, allocator)) == NULL) {
        throw std::runtime_error("xm_tensor_create");
    }
    xm_block_space_free(bsa);
    xm_block_space_free(bsb);
    xm_block_space_free(bsc);

    // setup tensor a
    setup_input_tensor<NA, T, bti_traits>(a, allocator, m_bta);
    // setup tensor b
    setup_input_tensor<NB, T, bti_traits>(b, allocator, m_btb);
    // setup tensor c
    setup_output_tensor_partial<NC, T, bti_traits>(c, allocator, btc, blklist);

    threading_policy::enable_omp_only();

    // contract
    double alpha = m_ka.get_coeff() * m_kb.get_coeff() * m_kc.get_coeff();
    xm_contract(alpha, a, b, 0.0, c, idxa, idxb, idxc);

    threading_policy::pop();

    // free all
    xm_tensor_free(a);
    xm_tensor_free(b);
    xm_tensor_free(c);
}

template<size_t N, size_t M, size_t K, typename T>
void bto_contract2_xm<N, M, K, T>::perform(
    gen_block_tensor_i<NC, bti_traits> &btc,
    const scalar_transf<T> &d) {

    typedef block_tensor_i_traits<T> bti_traits;

    gen_block_tensor_rd_ctrl<NC, bti_traits> cc(btc);
    std::vector<size_t> nzblkc;
    cc.req_nonzero_blocks(nzblkc);
    addition_schedule<NC, bto_traits<T> > asch(get_symmetry(),
        cc.req_const_symmetry());
    asch.build(get_schedule(), nzblkc);

    gen_bto_aux_add<NC, bto_traits<T> > out(get_symmetry(), asch, btc, d);
    out.open();
    perform(out);
    out.close();
}

template<size_t N, size_t M, size_t K, typename T>
void bto_contract2_xm<N, M, K, T>::perform(
    block_tensor_i<NC, T> &btc,
    T d) {

    perform(btc, scalar_transf<T>(d));
}

template<size_t N, size_t M, size_t K, typename T>
void bto_contract2_xm<N, M, K, T>::compute_block(
    bool zero,
    const index<NC> &ic,
    const tensor_transf<NC, T> &trc,
    dense_tensor_wr_i<NC, T> &blkc) {

    m_gbto.compute_block(zero, ic, trc, blkc);
}

} // namespace libtensor

#endif // LIBTENSOR_BTO_CONTRACT2_XM_IMPL_H
