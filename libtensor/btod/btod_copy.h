#ifndef LIBTENSOR_BTOD_COPY_H
#define LIBTENSOR_BTOD_COPY_H

#include <cmath>
#include <map>
#include "../defs.h"
#include "../exception.h"
#include "../timings.h"
#include "../core/abs_index.h"
#include "bad_block_index_space.h"
#include "additive_btod.h"
#include "../not_implemented.h"

namespace libtensor {


/** \brief Makes a copy of a block %tensor, applying a permutation and
        a scaling coefficient
    \tparam N Tensor order.

    \ingroup libtensor_btod
 **/
template<size_t N>
class btod_copy : public additive_btod<N>, public timings< btod_copy<N> > {
public:
    static const char *k_clazz; //!< Class name

private:
    block_tensor_i<N, double> &m_bta; //!< Source block %tensor
    permutation<N> m_perm; //!< Permutation
    double m_c; //!< Scaling coefficient
    block_index_space<N> m_bis; //!< Block %index space of output
    dimensions<N> m_bidims; //!< Block %index dimensions
    symmetry<N, double> m_sym; //!< Symmetry of output
    assignment_schedule<N, double> m_sch;

public:
    //!    \name Construction and destruction
    //@{

    /** \brief Initializes the copy operation
        \param bt Source block %tensor.
        \param c Scaling coefficient.
     **/
    btod_copy(block_tensor_i<N, double> &bta, double c = 1.0);

    /** \brief Initializes the permuted copy operation
        \param bt Source block %tensor.
        \param p Permutation.
        \param c Scaling coefficient.
     **/
    btod_copy(block_tensor_i<N, double> &bta, const permutation<N> &p,
        double c = 1.0);

    /** \brief Virtual destructor
     **/
    virtual ~btod_copy() { }
    //@}

    //!    \name Implementation of
    //!        libtensor::direct_block_tensor_operation<N, double>
    //@{
    virtual const block_index_space<N> &get_bis() const {
        return m_bis;
    }

    virtual const symmetry<N, double> &get_symmetry() const {
        return m_sym;
    }

    using additive_btod<N>::perform;

    virtual void sync_on();
    virtual void sync_off();

    //@}

    //!    \name Implementation of libtensor::btod_additive<N>
    //@{
    virtual const assignment_schedule<N, double> &get_schedule() const {
        return m_sch;
    }
    //@}

protected:
    virtual void compute_block(bool zero, dense_tensor_i<N, double> &blk,
        const index<N> &ib, const tensor_transf<N, double> &tr, double c,
        cpu_pool &cpus);

private:
    static block_index_space<N> mk_bis(const block_index_space<N> &bis,
        const permutation<N> &perm);
    void make_schedule();

private:
    btod_copy(const btod_copy<N>&);
    btod_copy<N> &operator=(const btod_copy<N>&);

};


} // namespace libtensor


#ifndef LIBTENSOR_INSTANTIATE_TEMPLATES
#include "btod_copy_impl.h"
#endif // LIBTENSOR_INSTANTIATE_TEMPLATES


#endif // LIBTENSOR_BTOD_COPY_H
