#ifndef LIBTENSOR_BTOD_ADD_H
#define LIBTENSOR_BTOD_ADD_H

#include <cmath>
#include <list>
#include <new>
#include <vector>
#include <utility>
#include <libtensor/timings.h>
#include <libtensor/core/block_tensor_i.h>
#include <libtensor/core/block_tensor_ctrl.h>
#include <libtensor/block_tensor/bto/additive_bto.h>
#include <libtensor/block_tensor/btod/btod_traits.h>

namespace libtensor {


/** \brief Addition of multiple block tensors

    This block %tensor operation performs the addition of block tensors:
    \f[ B = c_1 \mathcal{P}_1 A_1 + c_2 \mathcal{P}_2 A_2 + \cdots \f]

    The operation must have at least one operand provided at the time of
    construction. Other operands are added afterwards and must agree in
    the dimensions and the block structure.

    \ingroup libtensor_btod
 **/
template<size_t N>
class btod_add :
    public additive_bto<N, btod_traits>,
    public timings< btod_add<N> > {

public:
    static const char *k_clazz; //!< Class name

private:
    typedef timings< btod_add<N> > timings_base;

private:
    typedef struct operand {
        block_tensor_i<N, double> &m_bt; //!< Block %tensor
        permutation<N> m_perm; //!< Permutation
        double m_c; //!< Scaling coefficient
        operand(block_tensor_i<N, double> &bt,
            const permutation<N> &perm, double c)
        : m_bt(bt), m_perm(perm), m_c(c) { };
    } operand_t;

    typedef struct {
        block_tensor_ctrl<N, double> *m_ctrl;
        index<N> m_idx;
        tensor_transf<N, double> m_tr;
    } arg_t;

    struct schrec {
        size_t iarg;
        index<N> idx;
        permutation<N> perm;
        double k;
    };

    block_index_space<N> m_bis; //!< Block %index space of the result
    dimensions<N> m_bidims; //!< Block %index dimensions
    symmetry<N, double> m_sym; //!< Symmetry of the result
    std::vector<operand_t*> m_ops; //!< Operand list
    mutable bool m_dirty_sch; //!< Whether the schedule is dirty
    mutable assignment_schedule<N, double> *m_sch; //!< Assignment schedule
    mutable std::multimap<size_t, schrec> m_op_sch; //!< Operation schedule

    typedef typename std::multimap<size_t, schrec>::const_iterator
        schiterator_t;

public:
    //!    \name Construction, destruction, initialization
    //@{

    /** \brief Initializes the addition operation
        \param bt First block %tensor in the series.
        \param c Scaling coefficient.
     **/
    btod_add(block_tensor_i<N, double> &bt, double c = 1.0);

    /** \brief Initializes the addition operation
        \param bt First block %tensor in the series.
        \param pb Permutation of the first %tensor.
        \param c Scaling coefficient.
     **/
    btod_add(block_tensor_i<N, double> &bt, const permutation<N> &p,
        double c = 1.0);

    /** \brief Virtual destructor
     **/
    virtual ~btod_add();

    /** \brief Adds an operand (block %tensor in the series)
        \param bt Block %tensor.
        \param c Scaling coefficient.
        \throw bad_parameter If the block %tensor has incompatible
            %dimensions or block structure.
        \throw out_of_memory If memory allocation fails.
     **/
    void add_op(block_tensor_i<N, double> &bt, double c = 1.0);

    /** \brief Adds an operand (block %tensor in the series)
        \param bt Block %tensor.
        \param perm Permutation of the block %tensor.
        \param c Scaling coefficient.
        \throw bad_parameter If the block %tensor has incompatible
            %dimensions or block structure.
        \throw out_of_memory If memory allocation fails.
     **/
    void add_op(block_tensor_i<N, double> &bt, const permutation<N> &perm,
        double c = 1.0);

    //@}

    //!    \name Implementation of
    //      libtensor::direct_block_tensor_operation<N, double>
    //@{

    virtual const block_index_space<N> &get_bis() const {
        return m_bis;
    }

    virtual const symmetry<N, double> &get_symmetry() const {
        return m_sym;
    }

    virtual assignment_schedule<N, double> &get_schedule() const {
        if(m_dirty_sch || m_sch == 0) make_schedule();
        return *m_sch;
    }

    virtual void sync_on();
    virtual void sync_off();

    virtual void compute_block(bool zero, dense_tensor_i<N, double> &blk,
        const index<N> &i, const tensor_transf<N, double> &tr, const double &c);

    using additive_bto<N, btod_traits>::perform;

    virtual void perform(bto_stream_i<N, btod_traits> &out);

    //@}

private:
    void compute_block(dense_tensor_i<N, double> &blkb,
        const std::pair<schiterator_t, schiterator_t> ipair, bool zero,
        const tensor_transf<N, double> &trb, double kb);

    void add_operand(block_tensor_i<N, double> &bt,
        const permutation<N> &perm, double c);

    void make_schedule() const;

private:
    btod_add(const btod_add<N>&);
    btod_add<N> &operator=(const btod_add<N>&);

};


} // namespace libtensor

#endif // LIBTENSOR_BTOD_ADD_H
