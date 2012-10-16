#ifndef LIBTENSOR_GEN_BTO_CONTRACT2_CLST_H
#define LIBTENSOR_GEN_BTO_CONTRACT2_CLST_H

#include <list>
#include <libtensor/core/contraction2.h>
#include <libtensor/core/orbit_list.h>
#include <libtensor/core/noncopyable.h>
#include "../gen_block_tensor_i.h"

namespace libtensor {


/** \brief Computes the list of block contractions required to compute
        a block in C (base class)

    \sa gen_bto_contract2, gen_bto_contract2_clst

    \ingroup libtensor_gen_bto
 **/
template<size_t N, size_t M, size_t K, typename Traits>
class gen_bto_contract2_clst_base : public noncopyable {
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
    struct contr_pair {
        size_t aia, aib;
        tensor_transf<NA, element_type> tra;
        tensor_transf<NB, element_type> trb;
        contr_pair(
            size_t aia_,
            size_t aib_,
            const tensor_transf<NA, element_type> &tra_,
            const tensor_transf<NB, element_type> &trb_) :
            aia(aia_), aib(aib_), tra(tra_), trb(trb_) { }
    };

    typedef std::list<contr_pair> contr_list;

private:
    contr_list m_clst; //!< List of contractions

public:
    /** \brief Default constructor
     **/
    gen_bto_contract2_clst_base() { }

    /** \brief Returns true if the list of contractions is empty
     **/
    bool is_empty() const {
        return m_clst.empty();
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

    \sa gen_bto_contract2

    \ingroup libtensor_gen_bto
 **/
template<size_t N, size_t M, size_t K, typename Traits>
class gen_bto_contract2_clst :
    public gen_bto_contract2_clst_base<N, M, K, Traits> {

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

    //! Block tensor interface traits
    typedef typename Traits::bti_traits bti_traits;

public:
    typedef typename gen_bto_contract2_clst_base<N, M, K, Traits>::contr_pair
        contr_pair;
    typedef typename gen_bto_contract2_clst_base<N, M, K, Traits>::contr_list
        contr_list;

private:
    contraction2<N, M, K> m_contr; //!< Contraction descriptor
    gen_block_tensor_rd_i<NA, bti_traits> &m_bta; //!< First block tensor (A)
    gen_block_tensor_rd_i<NB, bti_traits> &m_btb; //!< Second block tensor (B)
    const orbit_list<NA, element_type> &m_ola; //!< List of orbits in A
    const orbit_list<NB, element_type> &m_olb; //!< List of orbits in B
    dimensions<NA> m_bidimsa; //!< Block index dimensions (A)
    dimensions<NB> m_bidimsb; //!< Block index dimensions (B)
    dimensions<NC> m_bidimsc; //!< Block index dimensions (C)
    index<NC> m_ic; //!< Index in C

public:
    gen_bto_contract2_clst(
        const contraction2<N, M, K> &contr,
        gen_block_tensor_rd_i<NA, bti_traits> &bta,
        gen_block_tensor_rd_i<NB, bti_traits> &btb,
        const orbit_list<NA, element_type> &ola,
        const orbit_list<NB, element_type> &olb,
        const dimensions<NA> &bidimsa,
        const dimensions<NB> &bidimsb,
        const dimensions<NC> &bidimsc,
        const index<NC> &ic);

    void build_list(bool testzero);

protected:
    using gen_bto_contract2_clst_base<N, M, K, Traits>::coalesce;
    using gen_bto_contract2_clst_base<N, M, K, Traits>::merge;

};


/** \brief Computes the list of block contractions required to compute
        a block in C (specialized for direct product, K = 0)

    \sa gen_bto_contract2

    \ingroup libtensor_gen_bto
 **/
template<size_t N, size_t M, typename Traits>
class gen_bto_contract2_clst<N, M, 0, Traits> :
    public gen_bto_contract2_clst_base<N, M, 0, Traits> {

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

    //! Block tensor interface traits
    typedef typename Traits::bti_traits bti_traits;

public:
    typedef typename gen_bto_contract2_clst_base<N, M, 0, Traits>::contr_pair
        contr_pair;
    typedef typename gen_bto_contract2_clst_base<N, M, 0, Traits>::contr_list
        contr_list;

private:
    contraction2<N, M, 0> m_contr; //!< Contraction descriptor
    gen_block_tensor_rd_i<NA, bti_traits> &m_bta; //!< First block tensor (A)
    gen_block_tensor_rd_i<NB, bti_traits> &m_btb; //!< Second block tensor (B)
    const orbit_list<NA, element_type> &m_ola; //!< List of orbits in A
    const orbit_list<NB, element_type> &m_olb; //!< List of orbits in B
    dimensions<NA> m_bidimsa; //!< Block index dimensions (A)
    dimensions<NB> m_bidimsb; //!< Block index dimensions (B)
    dimensions<NC> m_bidimsc; //!< Block index dimensions (C)
    index<NC> m_ic; //!< Index in C

public:
    gen_bto_contract2_clst(
        const contraction2<N, M, 0> &contr,
        gen_block_tensor_rd_i<NA, bti_traits> &bta,
        gen_block_tensor_rd_i<NB, bti_traits> &btb,
        const orbit_list<NA, element_type> &ola,
        const orbit_list<NB, element_type> &olb,
        const dimensions<NA> &bidimsa,
        const dimensions<NB> &bidimsb,
        const dimensions<NC> &bidimsc,
        const index<NC> &ic);

    void build_list(bool testzero);

protected:
    using gen_bto_contract2_clst_base<N, M, 0, Traits>::coalesce;
    using gen_bto_contract2_clst_base<N, M, 0, Traits>::merge;

};


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_CONTRACT2_CLST_H
