#ifndef LIBTENSOR_GEN_BTO_CONTRACT2_CLST_H
#define LIBTENSOR_GEN_BTO_CONTRACT2_CLST_H

#include <list>
#include <libtensor/core/tensor_transf.h>

namespace libtensor {


/** \brief Contraction pair
    \tparam N Order of first tensor less degree of contraction.
    \tparam M Order of second tensor less degree of contraction.
    \tparam K Order of contraction.
    \tparam T Element type of block tensor data

    A contraction pair stores the block indexes and block transformations of
    two blocks which have to be contracted as part of a block tensor
    contraction operation.

    \ingroup libtensor_gen_bto
 **/
template<size_t N, size_t M, size_t K, typename T>
class gen_bto_contract2_pair {
public:
    enum {
        NA = N + K,
        NB = M + K
    };
public:
    //! Type of tensor elements
    typedef T element_type;

private:
    size_t m_aia, m_acia, m_aib, m_acib;
    tensor_transf<NA, element_type> m_tra;
    tensor_transf<NB, element_type> m_trb;

public:
    /** \brief Constructor
        \param aia Absolute index of block in A.
        \param tra Transformation of block in A.
        \param aib Absolute index of block in B.
        \param trb Transformation of block in B.
     **/
    gen_bto_contract2_pair(
        size_t aia, size_t acia, const tensor_transf<NA, element_type> &tra,
        size_t aib, size_t acib, const tensor_transf<NB, element_type> &trb) :

        m_aia(aia), m_acia(acia), m_aib(aib), m_acib(acib),
        m_tra(tra), m_trb(trb)
    { }

    //! \name Manipulators ("Setters")
    //@{

    /** \brief Sets absolute index of block in A
     **/
    void set_abs_index_a(size_t aia) { m_aia = aia; }

    /** \brief Sets absolute index of block in B
     **/
    void set_abs_index_b(size_t aib) { m_aib = aib; }

    /** \brief Write access to transformation of block in A
     **/
    tensor_transf<N + K, T> &get_transf_a() { return m_tra; }

    /** \brief Write access to transformation of block in A
     **/
    tensor_transf<M + K, T> &get_transf_b() { return m_trb; }

    //@}

    //! \name Read access functions ("Getters")
    //@{

    /** \brief Return the absolute canonical index of block in A
     **/
    size_t get_acindex_a() const {
        return m_acia;
    }

    /** \brief Return the absolute canonical index of block in B
     **/
    size_t get_acindex_b() const {
        return m_acib;
    }

    /** \brief Return the absolute index of block in A
     **/
    size_t get_aindex_a() const {
        return m_aia;
    }

    /** \brief Return the absolute index of block in B
     **/
    size_t get_aindex_b() const {
        return m_aib;
    }

    /** \brief Return the transformation of block in A
     **/
    const tensor_transf<N + K, T> &get_transf_a() const { return m_tra; }

    /** \brief Return the transformation of block in B
     **/
    const tensor_transf<M + K, T> &get_transf_b() const { return m_trb; }
};


/** \brief List of contraction pairs.
    \tparam N Order of first tensor less degree of contraction.
    \tparam M Order of second tensor less degree of contraction.
    \tparam K Order of contraction.
    \tparam T Element type of block tensor data

    Defines types of contraction pair and contraction pair list.

    \ingroup libtensor_gen_bto
 **/
template<size_t N, size_t M, size_t K, typename T>
struct gen_bto_contract2_clst {

    //! List element type
    typedef gen_bto_contract2_pair<N, M, K, T> pair_type;

    //! List type
    typedef std::list<pair_type> list_type;
};


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_CONTRACT2_CLST_H
