#ifndef LIBTENSOR_BTOD_CONTRACT2_H
#define LIBTENSOR_BTOD_CONTRACT2_H

#include <list>
#include <map>
#include <vector>
#include <libutil/threads/auto_lock.h>
#include <libutil/thread_pool/thread_pool.h>
#include "../defs.h"
#include "../exception.h"
#include "../timings.h"
#include "../core/abs_index.h"
#include "../core/block_tensor_i.h"
#include "../core/block_tensor_ctrl.h"
#include "../core/orbit.h"
#include "../core/orbit_list.h"
#include "../core/sequence.h"
#include "../tod/contraction2.h"
#include <libtensor/block_tensor/bto/additive_bto.h>
#include <libtensor/block_tensor/btod/btod_traits.h>
#include <libtensor/block_tensor/bto/bto_contract2_sym.h>
#include "../not_implemented.h"
#include "bad_block_index_space.h"
#include <libtensor/core/scalar_transf_double.h>

namespace libtensor {

template<size_t N, size_t M, size_t K>
class btod_contract2_symmetry_builder_base;

template<size_t N, size_t M, size_t K> class btod_contract2_symmetry_builder;

template<size_t N, size_t K> class btod_contract2_symmetry_builder<N, N, K>;

template<size_t N, size_t M, size_t K>
struct btod_contract2_clazz {
    static const char *k_clazz;
};


/** \brief Contraction of two block tensors

    \ingroup libtensor_btod
 **/
template<size_t N, size_t M, size_t K>
class btod_contract2 :
    public additive_bto<N + M, bto_traits<double> >,
    public timings< btod_contract2<N, M, K> > {

public:
    static const char *k_clazz; //!< Class name

private:
    enum {
        k_ordera = N + K, //!< Order of first argument (A)
        k_orderb = M + K, //!< Order of second argument (B)
        k_orderc = N + M, //!< Order of result (C)
        k_totidx = N + M + K, //!< Total number of indexes
        k_maxconn = 2 * k_totidx, //!< Index connections
    };

private:
    typedef struct block_contr {
    public:
        size_t m_absidxa;
        size_t m_absidxb;
        double m_c;
        permutation<k_ordera> m_perma;
        permutation<k_orderb> m_permb;

    public:
        block_contr(size_t aia, size_t aib, double c,
                    const permutation<k_ordera> &perma,
                    const permutation<k_orderb> &permb)
            : m_absidxa(aia), m_absidxb(aib), m_c(c), m_perma(perma),
              m_permb(permb)
            { }
        bool is_same_perm(const tensor_transf<k_ordera, double> &tra,
                          const tensor_transf<k_orderb, double> &trb) {

            return m_perma.equals(tra.get_perm()) &&
                m_permb.equals(trb.get_perm());
        }
    } block_contr_t;
    typedef std::list<block_contr_t> block_contr_list_t;
    typedef typename std::list<block_contr_t>::iterator 
        block_contr_list_iterator_t;
    typedef std::pair<block_contr_list_t, volatile bool>
    block_contr_list_pair_t;
    typedef std::map<size_t, block_contr_list_pair_t*> schedule_t;

    class make_schedule_task :
        public libutil::task_i,
        public timings<make_schedule_task> {

    public:
        static const char *k_clazz;

    private:
        contraction2<N, M, K> m_contr;
        block_tensor_ctrl<k_ordera, double> m_ca;
        block_tensor_ctrl<k_orderb, double> m_cb;
        const symmetry<k_orderc, double> &m_symc;
        dimensions<k_ordera> m_bidimsa;
        dimensions<k_orderb> m_bidimsb;
        dimensions<k_orderc> m_bidimsc;
        const orbit_list<k_ordera, double> &m_ola;
        typename orbit_list<k_ordera, double>::iterator m_ioa1;
        typename orbit_list<k_ordera, double>::iterator m_ioa2;
        schedule_t &m_contr_sch;
        schedule_t m_contr_sch_local;
        assignment_schedule<k_orderc, double> &m_sch;
        libutil::mutex &m_sch_lock;

    public:
        make_schedule_task(const contraction2<N, M, K> &contr,
                           block_tensor_i<k_ordera, double> &bta,
                           block_tensor_i<k_orderb, double> &btb,
                           const symmetry<k_orderc, double> &symc,
                           const dimensions<k_orderc> &bidimsc,
                           const orbit_list<k_ordera, double> &ola,
                           const typename orbit_list<k_ordera,
                           double>::iterator &ioa1,
                           const typename orbit_list<k_ordera,
                           double>::iterator &ioa2,
                           schedule_t &contr_sch,
                           assignment_schedule<k_orderc, double> &sch,
                           libutil::mutex &sch_lock);
        virtual ~make_schedule_task() { }
        virtual void perform();

    private:
        void make_schedule_a(const orbit_list<k_orderc, double> &olc,
                             const abs_index<k_ordera> &aia,
                             const abs_index<k_ordera> &acia,
                             const tensor_transf<k_ordera, double> &tra);
        void make_schedule_b(const abs_index<k_ordera> &acia,
                             const tensor_transf<k_ordera, double> &tra,
                             const index<k_orderb> &ib,
                             const abs_index<k_orderc> &acic);
        void schedule_block_contraction(const abs_index<k_orderc> &acic,
                                        const block_contr_t &bc);
        void merge_schedule();
        void merge_lists(const block_contr_list_t &src,
                         block_contr_list_t &dst);
        block_contr_list_iterator_t merge_node(
            const block_contr_t &bc, block_contr_list_t &lst,
            const block_contr_list_iterator_t &begin);
    };

    class make_schedule_task_iterator : public libutil::task_iterator_i {
    private:
        std::vector<make_schedule_task*> &m_tl;
        typename std::vector<make_schedule_task*>::iterator m_i;
    public:
        make_schedule_task_iterator(std::vector<make_schedule_task*> &tl) :
            m_tl(tl), m_i(m_tl.begin()) { }
        virtual bool has_more() const;
        virtual libutil::task_i *get_next();
    };

    class make_schedule_task_observer : public libutil::task_observer_i {
    public:
        virtual void notify_start_task(libutil::task_i *t) { }
        virtual void notify_finish_task(libutil::task_i *t) { }
    };

private:
    contraction2<N, M, K> m_contr; //!< Contraction
    block_tensor_i<k_ordera, double> &m_bta; //!< First argument (A)
    block_tensor_i<k_orderb, double> &m_btb; //!< Second argument (B)
    bto_contract2_sym<N, M, K, double> m_symc; //!< Symmetry of result (C)
    dimensions<k_ordera> m_bidimsa; //!< Block %index dims of A
    dimensions<k_orderb> m_bidimsb; //!< Block %index dims of B
    dimensions<k_orderc> m_bidimsc; //!< Block %index dims of the result
    schedule_t m_contr_sch; //!< Contraction schedule
    assignment_schedule<k_orderc, double> m_sch; //!< Assignment schedule

public:
    //!    \name Construction and destruction
    //@{

    /** \brief Initializes the contraction operation
        \param contr Contraction.
        \param bta Block %tensor A (first argument).
        \param btb Block %tensor B (second argument).
    **/
    btod_contract2(const contraction2<N, M, K> &contr,
                   block_tensor_i<k_ordera, double> &bta,
                   block_tensor_i<k_orderb, double> &btb);

    /** \brief Virtual destructor
     **/
    virtual ~btod_contract2();

    //@}

    //!    \name Implementation of
    //      libtensor::direct_block_tensor_operation<N + M, double>
    //@{

    virtual const block_index_space<N + M> &get_bis() const;
    virtual const symmetry<N + M, double> &get_symmetry() const;
    virtual const assignment_schedule<N + M, double> &get_schedule() const;
    virtual void sync_on();
    virtual void sync_off();

    //@}

    virtual void perform(block_tensor_i<N + M, double> &btc);
    virtual void perform(block_tensor_i<N + M, double> &btc, double d);

    virtual void compute_block(bool zero, dense_tensor_i<N + M, double> &blk,
        const index<N + M> &i, const tensor_transf<N + M, double> &tr,
        const double &c);

private:
    void perform_inner(block_tensor_i<N + M, double> &btc, double d,
        const std::vector<size_t> &blst);

    void make_schedule();

    void clear_schedule(schedule_t &sch);

    void contract_block(
        block_contr_list_t &lst, const index<k_orderc> &idxc,
        block_tensor_ctrl<k_ordera, double> &ctrla,
        block_tensor_ctrl<k_orderb, double> &ctrlb,
        dense_tensor_i<k_orderc, double> &blkc,
        const tensor_transf<k_orderc, double> &trc,
        bool zero, double c);

private:
    btod_contract2(const btod_contract2<N, M, K>&);
    btod_contract2<N, M, K> &operator=(const btod_contract2<N, M, K>&);

};


} // namespace libtensor

#endif // LIBTENSOR_BTOD_CONTRACT2_H
