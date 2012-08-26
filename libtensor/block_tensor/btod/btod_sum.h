#ifndef LIBTENSOR_BTOD_SUM_H
#define LIBTENSOR_BTOD_SUM_H

#include <list>
#include <libtensor/timings.h>
#include <libtensor/block_tensor/bto/additive_bto.h>
#include <libtensor/block_tensor/btod/btod_traits.h>

namespace libtensor {


/** \brief Adds results of a %sequence of block %tensor operations
        (for double)
    \tparam N Tensor order.

    This operation runs a %sequence of block %tensor operations and
    accumulates their results with given coefficients. All of the operations
    in the %sequence shall derive from additive_bto<N, btod_traits>.

    The %sequence must contain at least one operation, which is called the
    base operation.

    \ingroup libtensor_btod
 **/
template<size_t N>
class btod_sum :
    public additive_bto<N, btod_traits>,
    public timings< btod_sum<N> > {

public:
    static const char* k_clazz; //!< Class name

private:
    //!    \brief List node type
    typedef struct node {
    private:
        additive_bto<N, btod_traits> *m_op;
       	double m_c;
    public:
        node() : m_op(NULL), m_c(0.0) { }
        node(additive_bto<N, btod_traits> &op, double c) : m_op(&op), m_c(c) { }
        additive_bto<N, btod_traits> &get_op() { return *m_op; }
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
    btod_sum(additive_bto<N, btod_traits> &op, double c = 1.0);

    /** \brief Virtual destructor
     **/
    virtual ~btod_sum();

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

    virtual void sync_on();
    virtual void sync_off();

    //@}


    //!    \name Implementation of libtensor::additive_bto<N, btod_traits>
    //@{

    using additive_bto<N, btod_traits>::compute_block;
    virtual void compute_block(bool zero, dense_tensor_i<N, double> &blk,
        const index<N> &i, const tensor_transf<N, double> &tr, const double &c);
    virtual void perform(block_tensor_i<N, double> &btb);
    virtual void perform(block_tensor_i<N, double> &btb, const double &c);
    virtual void perform(bto_stream_i<N, btod_traits> &out);
    virtual void perform(block_tensor_i<N, double> &btb, const double &c,
        const std::vector<size_t> &blst) {
        additive_bto<N, btod_traits>::perform(btb, c, blst);
    }

    //@}


    //!    \name Manipulations
    //@{

    /** \brief Adds an operation to the sequence
        \param op Operation.
        \param c Coefficient.
     **/
    void add_op(additive_bto<N, btod_traits> &op, double c = 1.0);

    //@}

private:
    void make_schedule() const;

private:
    btod_sum<N> &operator=(const btod_sum<N>&);

};


} // namespace libtensor

#endif // LIBTENSOR_BTOD_SUM_H

