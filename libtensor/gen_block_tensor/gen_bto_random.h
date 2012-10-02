#ifndef LIBTENSOR_GEN_BTO_RANDOM_H
#define LIBTENSOR_GEN_BTO_RANDOM_H

#include <list>
#include <map>
#include <libtensor/defs.h>
#include <libtensor/timings.h>
#include <libtensor/exception.h>
#include <libtensor/core/dimensions.h>
#include <libtensor/core/noncopyable.h>
#include "gen_block_tensor_i.h"
#include "gen_block_tensor_ctrl.h"

namespace libtensor {


/** \brief Fills a block %tensor with random data without affecting its
        %symmetry
    \tparam T Block %tensor order.

    \ingroup libtensor_btod
 **/
template<size_t N, typename Traits, typename Timed>
class gen_bto_random : public timings<Timed>, public noncopyable {
public:
    //! Type of tensor elements
    typedef typename Traits::element_type element_type;

    //! Block tensor interface traits
    typedef typename Traits::bti_traits bti_traits;

    //! Type of read-only block
    typedef typename bti_traits::template rd_block_type<N>::type rd_block_type;

    //! Type of write-only block
    typedef typename bti_traits::template wr_block_type<N>::type wr_block_type;

    //! Type of tensor transformation
    typedef tensor_transf<N, element_type> tensor_transf_type;

private:
    typedef std::list<tensor_transf_type> transf_list_t;
    typedef std::map<size_t, transf_list_t> transf_map_t;

public:
    static const char *k_clazz; //!< Class name

public:
    /** \brief Fills a block %tensor with random values preserving
            symmetry
        \param bt Block %tensor.
     **/
    void perform(gen_block_tensor_wr_i<N, bti_traits> &bt) throw(exception);

    /** \brief Fills one block of a block %tensor with random values
            preserving symmetry
        \param bt Block %tensor.
        \param idx Block %index in the block %tensor.
     **/
    void perform(gen_block_tensor_wr_i<N, bti_traits> &bt,
            const index<N> &idx) throw(exception);

private:
    bool make_transf_map(const symmetry<N, element_type> &sym,
        const dimensions<N> &bidims, const index<N> &idx,
        const tensor_transf_type &tr, transf_map_t &alltransf);

    void make_random_blk(gen_block_tensor_wr_ctrl<N, bti_traits> &ctrl,
        const dimensions<N> &bidims, const index<N> &idx);
};


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_RANDOM_H
