#ifndef LIBTENSOR_BTOD_SYMMETRIZE3_H
#define LIBTENSOR_BTOD_SYMMETRIZE3_H

#include <map>
#include <libtensor/timings.h>
#include <libtensor/block_tensor/btod/btod_traits.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/gen_block_tensor/additive_gen_bto.h>

namespace libtensor {


/** \brief (Anti-)symmetrizes the result of a block %tensor operation
        over three groups of indexes
    \tparam N Tensor order.

    The operation symmetrizes or anti-symmetrizes the result of another
    block %tensor operation over three indexes or groups of indexes.

    \f[
        b_{ijk} = P_{\pm} a_{ijk} = a_{ijk} \pm a_{jik} \pm a_{kji} \pm
            a_{ikj} + a_{jki} + a_{kij}
    \f]

    The constructor takes three different indexes to be symmetrized.

    \ingroup libtensor_btod
 **/
template<size_t N>
class btod_symmetrize3 :
    public additive_gen_bto<N, btod_traits::bti_traits>,
    public timings< btod_symmetrize3<N> >,
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
    size_t m_i1; //!< First %index
    size_t m_i2; //!< Second %index
    size_t m_i3; //!< Third %index
    bool m_symm; //!< Symmetrization/anti-symmetrization
    symmetry<N, double> m_sym; //!< Symmetry of the result
    assignment_schedule<N, double> m_sch; //!< Schedule

public:
    //!    \name Construction and destruction
    //@{

    /** \brief Initializes the operation
        \param op Operation to be symmetrized.
        \param i1 First %index.
        \param i2 Second %index.
        \param i3 Third %index.
        \param symm True for symmetrization, false for
            anti-symmetrization.
     **/
    btod_symmetrize3(additive_gen_bto<N, bti_traits> &op,
            size_t i1, size_t i2, size_t i3, bool symm);

    /** \brief Virtual destructor
     **/
    virtual ~btod_symmetrize3() { }

    //@}


    //!    \name Implementation of direct_gen_bto<N, bti_traits>
    //@{

    virtual const block_index_space<N> &get_bis() const {
        return m_op.get_bis();
    }

    virtual const symmetry<N, double> &get_symmetry() const {
        return m_sym;
    }

    virtual const assignment_schedule<N, double> &get_schedule() const {
        return m_sch;
    }

    virtual void perform(gen_block_stream_i<N, bti_traits> &out);
    virtual void perform(gen_block_tensor_i<N, bti_traits> &btc);
    virtual void perform(gen_block_tensor_i<N, bti_traits> &btc,
            const scalar_transf<double> &d);

    //@}

    virtual void perform(block_tensor_i<N, double> &btc, double d);

protected:
    //!    \brief Implementation of additive_bto<N, btod_traits>
    //@{

    virtual void compute_block(
            bool zero,
            const index<N> &i,
            const tensor_transf<N, double> &tr,
            dense_tensor_wr_i<N, double> &blk);

    //@}

private:
    void make_symmetry();
    void make_schedule();
    void make_schedule_blk(const abs_index<N> &ai,
        sym_schedule_t &sch) const;
};


} // namespace libtensor

#endif // LIBTENSOR_BTOD_SYMMETRIZE3_H
