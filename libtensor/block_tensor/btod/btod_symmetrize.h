#ifndef LIBTENSOR_BTOD_SYMMETRIZE_H
#define LIBTENSOR_BTOD_SYMMETRIZE_H

#include <map>
#include <libtensor/timings.h>
#include <libtensor/block_tensor/btod/btod_traits.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/gen_block_tensor/additive_gen_bto.h>

namespace libtensor {


/** \brief Symmetrizes the result of another block %tensor operation
    \tparam N Tensor order.

    \ingroup libtensor_btod
 **/
template<size_t N>
class btod_symmetrize :
    public additive_gen_bto<N, btod_traits::bti_traits>,
    public timings< btod_symmetrize<N> >,
    public noncopyable {

public:
    static const char *k_clazz; //!< Class name

public:
    typedef typename btod_traits::bti_traits bti_traits;

private:
    struct schrec {
        size_t ai;
        tensor_transf<N, double> tr;
        schrec() : ai(0) { }
        schrec(size_t ai_, const tensor_transf<N, double> &tr_) :
            ai(ai_), tr(tr_) { }
    };
    typedef std::pair<size_t, schrec> sym_schedule_pair_t;
    typedef std::multimap<size_t, schrec> sym_schedule_t;

private:
    additive_gen_bto<N, bti_traits> &m_op; //!< Symmetrized operation
    permutation<N> m_perm1; //!< First symmetrization permutation
    bool m_symm; //!< Symmetrization sign
    block_index_space<N> m_bis; //!< Block %index space of the result
    symmetry<N, double> m_sym; //!< Symmetry of the result
    assignment_schedule<N, double> m_sch; //!< Schedule
    sym_schedule_t m_sym_sch; //!< Symmetrization schedule

public:
    //!    \name Construction and destruction
    //@{

    /** \brief Initializes the operation to symmetrize two indexes
        \param op Symmetrized operation.
        \param i1 First %tensor %index.
        \param i2 Second %tensor %index.
        \param symm True for symmetric, false for anti-symmetric.
     **/
    btod_symmetrize(additive_gen_bto<N, bti_traits> &op,
            size_t i1, size_t i2, bool symm);

    /** \brief Initializes the operation using a unitary %permutation
            (P = P^-1)
        \param op Symmetrized operation.
        \param perm Unitary %permutation.
        \param symm True for symmetric, false for anti-symmetric.
     **/
    btod_symmetrize(additive_gen_bto<N, bti_traits> &op,
            const permutation<N> &perm, bool symm);

    /** \brief Virtual destructor
     **/
    virtual ~btod_symmetrize() { }

    //@}


    //!    \name Implementation of direct_gen_bto<N, bti_traits>
    //@{

    virtual const block_index_space<N> &get_bis() const {
        return m_bis;
    }

    virtual const symmetry<N, double> &get_symmetry() const {
        return m_sym;
    }

    virtual const assignment_schedule<N, double> &get_schedule() const {
        return m_sch;
    }

    virtual void perform(gen_block_stream_i<N, bti_traits> &out);

    //@}

    virtual void perform(gen_block_tensor_i<N, bti_traits> &btc);
    virtual void perform(gen_block_tensor_i<N, bti_traits> &btc,
            const scalar_transf<double> &d);

    virtual void perform(block_tensor_i<N, double> &btc, double d);

protected:
    //!    \brief Implementation of additive_bto<N, btod_traits>
    //@{

    virtual void compute_block(
            bool zero,
            const index<N> &i,
            const tensor_transf<N, double> &tr,
            dense_tensor_i<N, double> &blk);

    //@}

private:
    /** \brief Constructs the %symmetry of the result
     **/
    void make_symmetry();

    /** \brief Constructs the assignment schedule of the operation
     **/
    void make_schedule();
};


} // namespace libtensor

#endif // LIBTENSOR_BTOD_SYMMETRIZE_H
