#ifndef LIBTENSOR_GEN_BTO_CONTRACT2_CLST_BUILDER_H
#define LIBTENSOR_GEN_BTO_CONTRACT2_CLST_BUILDER_H

#include <list>
#include <libtensor/core/contraction2.h>
#include <libtensor/core/orbit_list.h>
#include <libtensor/core/noncopyable.h>
#include "../gen_block_tensor_i.h"
#include "../gen_bto_contract2_clst.h"
#include "block_list.h"
#include "gen_bto_contract2_block_list.h"

namespace libtensor {


/** \brief Computes the list of block contractions required to compute
        a block in C (base class)
    \tparam N Order of first tensor less degree of contraction.
    \tparam M Order of second tensor less degree of contraction.
    \tparam K Order of contraction.
    \tparam Traits Traits class.

    The traits class has to provide definitions for
    - \c element_type -- Type of data elements
    - \c template to_contract2_type<N, M, K>::clst_optimize_type -- Type of
            contraction pair list optimizer (\sa gen_bto_contract2_clst_builder)

    \sa gen_bto_contract2_block, gen_bto_contract2_clst

    \ingroup libtensor_gen_bto
 **/
template<size_t N, size_t M, size_t K, typename Traits>
class gen_bto_contract2_clst_builder_base : public noncopyable {
public:
    enum {
        NA = N + K, //!< Order of first argument (A)
        NB = M + K, //!< Order of second argument (B)
        NC = N + M  //!< Order of result (C)
    };

public:
    //! Type of tensor elements
    typedef typename Traits::element_type element_type;
    typedef typename gen_bto_contract2_clst<N, M, K, element_type>::list_type
            contr_list;

private:
    contraction2<N, M, K> m_contr; //!< Contraction
    contr_list m_clst; //!< List of contractions

public:
    /** \brief Initializes the builder
     **/
    gen_bto_contract2_clst_builder_base(const contraction2<N, M, K> &contr) :
        m_contr(contr)
    { }

    /** \brief Returns true if the list of contractions is empty
     **/
    bool is_empty() const {
        return m_clst.empty();
    }

    /** \brief Returns the contraction
     **/
    const contraction2<N, M, K> &get_contr() const {
        return m_contr;
    }

    /** \brief Returns the list of contractions
     **/
    const contr_list &get_clst() const {
        return m_clst;
    }

protected:
    void coalesce(contr_list &clst);
    void merge(contr_list &clst);

};


/** \brief Computes the list of block contractions required to compute
        a block in C
    \tparam N Order of first tensor less degree of contraction.
    \tparam M Order of second tensor less degree of contraction.
    \tparam K Order of contraction.
    \tparam Traits Traits class.

    \sa gen_bto_contract2_clst_builder_base, gen_bto_unfold_block_list

    \ingroup libtensor_gen_bto
 **/
template<size_t N, size_t M, size_t K, typename Traits>
class gen_bto_contract2_clst_builder :
    public gen_bto_contract2_clst_builder_base<N, M, K, Traits> {

public:
    static const char *k_clazz; //!< Class name

public:
    enum {
        NA = N + K, //!< Order of first argument (A)
        NB = M + K, //!< Order of second argument (B)
        NC = N + M  //!< Order of result (C)
    };

public:
    //! Type of tensor elements
    typedef typename Traits::element_type element_type;

public:
    typedef typename gen_bto_contract2_clst<N, M, K, element_type>::pair_type
            contr_pair;
    typedef typename gen_bto_contract2_clst<N, M, K, element_type>::list_type
            contr_list;

private:
    const symmetry<NA, element_type> &m_syma; //!< Symmetry of A
    const symmetry<NB, element_type> &m_symb; //!< Symmetry of B
    const block_list<NA> &m_blka; //!< All non-zero blocks in A
    const block_list<NB> &m_blkb; //!< All non-zero blocks in B
    dimensions<NC> m_bidimsc; //!< Block index dimensions (C)
    index<NC> m_ic; //!< Index in C

public:
    gen_bto_contract2_clst_builder(
        const contraction2<N, M, K> &contr,
        const symmetry<NA, element_type> &syma,
        const symmetry<NB, element_type> &symb,
        const block_list<NA> &blka,
        const block_list<NB> &blkb,
        const dimensions<NC> &bidimsc,
        const index<NC> &ic);

    void build_list(bool testzero);
    void build_list(bool testzero,
        gen_bto_contract2_block_list<N, M, K, Traits> &bl);

private:
    void build_list_2(
        gen_bto_contract2_block_list<N, M, K, Traits> &bl,
        contr_list &clst);

protected:
    using gen_bto_contract2_clst_builder_base<N, M, K, Traits>::get_contr;
    using gen_bto_contract2_clst_builder_base<N, M, K, Traits>::coalesce;
    using gen_bto_contract2_clst_builder_base<N, M, K, Traits>::merge;

};


/** \brief Computes the list of block contractions required to compute
        a block in C (specialized for direct product, K = 0)
    \tparam N Order of first tensor less degree of contraction.
    \tparam M Order of second tensor less degree of contraction.
    \tparam Traits Traits class.

    \sa gen_bto_contract2_clst_builder_base, gen_bto_unfold_block_list

    \ingroup libtensor_gen_bto
 **/
template<size_t N, size_t M, typename Traits>
class gen_bto_contract2_clst_builder<N, M, 0, Traits> :
    public gen_bto_contract2_clst_builder_base<N, M, 0, Traits> {

public:
    static const char *k_clazz; //!< Class name

public:
    enum {
        NA = N, //!< Order of first argument (A)
        NB = M, //!< Order of second argument (B)
        NC = N + M  //!< Order of result (C)
    };

public:
    //! Type of tensor elements
    typedef typename Traits::element_type element_type;

public:
    typedef typename gen_bto_contract2_clst<N, M, 0, element_type>::pair_type
            contr_pair;
    typedef typename gen_bto_contract2_clst<N, M, 0, element_type>::list_type
            contr_list;

private:
    const symmetry<NA, element_type> &m_syma; //!< Symmetry of A
    const symmetry<NB, element_type> &m_symb; //!< Symmetry of B
    const block_list<NA> &m_blka; //!< All non-zero blocks in A
    const block_list<NB> &m_blkb; //!< All non-zero blocks in B
    dimensions<NC> m_bidimsc; //!< Block index dimensions (C)
    index<NC> m_ic; //!< Index in C

public:
    gen_bto_contract2_clst_builder(
        const contraction2<N, M, 0> &contr,
        const symmetry<NA, element_type> &syma,
        const symmetry<NB, element_type> &symb,
        const block_list<NA> &blka,
        const block_list<NB> &blkb,
        const dimensions<NC> &bidimsc,
        const index<NC> &ic);

    void build_list(bool testzero);
    void build_list(bool testzero,
        gen_bto_contract2_block_list<N, M, 0, Traits> &bl) {
        build_list(testzero);
    }

protected:
    using gen_bto_contract2_clst_builder_base<N, M, 0, Traits>::get_contr;
    using gen_bto_contract2_clst_builder_base<N, M, 0, Traits>::coalesce;
    using gen_bto_contract2_clst_builder_base<N, M, 0, Traits>::merge;

};


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_CONTRACT2_CLST_BUILDER_H
