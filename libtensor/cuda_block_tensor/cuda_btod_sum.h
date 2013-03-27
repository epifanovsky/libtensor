#ifndef LIBTENSOR_CUDA_BTOD_SUM_H
#define LIBTENSOR_CUDA_BTOD_SUM_H

#include <list>
#include <libtensor/timings.h>
#include <libtensor/cuda_block_tensor/cuda_btod_traits.h>
#include <libtensor/gen_block_tensor/additive_gen_bto.h>
#include <libtensor/cuda/cuda_allocator.h>

namespace libtensor {


/** \brief Adds results of a %sequence of block %tensor operations
        (for double)
    \tparam N Tensor order.

    This operation runs a %sequence of block %tensor operations and
    accumulates their results with given coefficients. All of the operations
    in the %sequence shall derive from additive_bto<N, cuda_btod_traits>.

    The %sequence must contain at least one operation, which is called the
    base operation.

    Marked deprecated!!! Will be removed when interface has been generalized.

    \ingroup libtensor_btod
 **/
template<size_t N>
class cuda_btod_sum :
    public additive_gen_bto<N, cuda_btod_traits::bti_traits>,
    public timings< cuda_btod_sum<N> > {

typedef cuda_allocator<double> cuda_allocator_t;

public:
    static const char* k_clazz; //!< Class name

public:
    typedef typename cuda_btod_traits::bti_traits bti_traits;

private:
    //!    \brief List node type
    typedef struct node {
    private:
        additive_gen_bto<N, bti_traits> *m_op;
        double m_c;
    public:
        node() : m_op(NULL), m_c(0.0) { }
        node(additive_gen_bto<N, bti_traits> &op, double c) :
            m_op(&op), m_c(c) { }

        additive_gen_bto<N, bti_traits> &get_op() { return *m_op; }

        double get_coeff() const { return m_c; }
    } node_t;

private:
    mutable std::list<node_t> m_ops; //!< List of operations
    block_index_space<N> m_bis; //!< Block index space
    dimensions<N> m_bidims; //!< Block index dims
    symmetry<N, double> m_sym; //!< Symmetry of operation
    mutable bool m_dirty_sch; //!< Whether the assignment schedule is dirty
    mutable assignment_schedule<N, double> *m_sch; //!< Assignment schedule

public:
    //!    \name Construction and destruction
    //@{

    /** \brief Initializes the base operation
        \param op Operation.
        \param c Coefficient.
     **/
    cuda_btod_sum(additive_gen_bto<N, bti_traits> &op, double c = 1.0);

    /** \brief Virtual destructor
     **/
    virtual ~cuda_btod_sum();

    //@}


    //!    \name Implementation of libtensor::direct_tensor_operation<N>
    //@{

    virtual const block_index_space<N> &get_bis() const {
        return m_bis;
    }

    virtual const symmetry<N, double> &get_symmetry() const {
        return m_sym;
    }

    virtual const assignment_schedule<N, double> &get_schedule() const {
        if(m_sch == 0 || m_dirty_sch) make_schedule();
        return *m_sch;
    }

    //@}


    //!    \name Implementation of libtensor::additive_bto<N, cuda_btod_traits>
    //@{

    virtual void compute_block(
            bool zero,
            const index<N> &i,
            const tensor_transf<N, double> &tr,
            dense_tensor_wr_i<N, double> &blk);

    virtual void perform(gen_block_tensor_i<N, bti_traits> &btb);
    virtual void perform(gen_block_tensor_i<N, bti_traits> &btb,
            const scalar_transf<double> &c);
    virtual void perform(gen_block_stream_i<N, bti_traits> &out);

    //@}

    void perform(gen_block_tensor_i<N, bti_traits> &btb, double c);

    //!    \name Manipulations
    //@{

    /** \brief Adds an operation to the sequence
        \param op Operation.
        \param c Coefficient.
     **/
    void add_op(additive_gen_bto<N, bti_traits> &op, double c = 1.0);

    //@}

private:
    void make_schedule() const;

private:
    cuda_btod_sum<N> &operator=(const cuda_btod_sum<N>&);

};


} // namespace libtensor

#endif // LIBTENSOR_CUDA_BTOD_SUM_H

