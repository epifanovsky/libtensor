#ifndef LIBTENSOR_CTF_BTOD_SET_SYMMETRY_H
#define LIBTENSOR_CTF_BTOD_SET_SYMMETRY_H

#include <libtensor/core/noncopyable.h>
#include <libtensor/gen_block_tensor/addition_schedule.h>
#include "../ctf_btod_traits.h"

namespace libtensor {


/** \brief Sets the permutational symmetry of distributed blocks in accordance
        with the symmetry of the block tensor
    \tparam N Tensor order.

    \ingroup libtensor_ctf_block_tensor
 **/
template<size_t N>
class ctf_btod_set_symmetry : public noncopyable {
public:
    static const char k_clazz[]; //!< Class name

public:
    typedef ctf_block_tensor_i_traits<double> bti_traits;

public:
    /** \brief Performs the operation
        \param blst List of blocks
        \param bt Output block tensor
     **/
    void perform(
        const std::vector<size_t> &blst,
        gen_block_tensor_i<N, bti_traits> &bt);

    /** \brief Performs the operation
        \param sch Assignment schedule
        \param bt Output block tensor
     **/
    void perform(
        const assignment_schedule<N, double> &sch,
        gen_block_tensor_i<N, bti_traits> &bt);

    /** \brief Performs the operation
        \param asch Addition schedule
        \param bt Output block tensor
     **/
    void perform(
        const addition_schedule<N, ctf_btod_traits> &asch,
        gen_block_tensor_i<N, bti_traits> &bt);

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_BTOD_SET_SYMMETRY_H
