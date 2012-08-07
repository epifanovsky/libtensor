#ifndef LIBTENSOR_BTO_CONTRACT2_CLST_H
#define LIBTENSOR_BTO_CONTRACT2_CLST_H

#include <list>
#include <libtensor/core/block_tensor_i.h>
#include <libtensor/core/orbit_list.h>
#include <libtensor/tod/contraction2.h>

namespace libtensor {


/** \brief Computes the list of block contractions required to compute
        a block in C (base class)

    \sa bto_contract2

    \ingroup ccman2_block_tensor_btod
 **/
template<size_t N, size_t M, size_t K, typename T>
class bto_contract2_clst_base {
public:
    struct contr_pair {
        size_t aia, aib;
        tensor_transf<N + K, T> tra;
        tensor_transf<M + K, T> trb;
        contr_pair(
            size_t aia_,
            size_t aib_,
            const tensor_transf<N + K, T> &tra_,
            const tensor_transf<M + K, T> &trb_) :
            aia(aia_), aib(aib_), tra(tra_), trb(trb_) { }
    };

    typedef std::list<contr_pair> contr_list;

private:
    contr_list m_clst; //!< List of contractions

public:
    /** \brief Default constructor
     **/
    bto_contract2_clst_base() { }

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

private:
    /** \brief Private copy constructor
     **/
    bto_contract2_clst_base(const bto_contract2_clst_base&);

};


/** \brief Computes the list of block contractions required to compute
        a block in C

    \sa bto_contract2

    \ingroup ccman2_block_tensor_btod
 **/
template<size_t N, size_t M, size_t K, typename T>
class bto_contract2_clst : public bto_contract2_clst_base<N, M, K, T> {
public:
    static const char *k_clazz; //!< Class name

public:
    typedef typename bto_contract2_clst_base<N, M, K, T>::contr_pair contr_pair;
    typedef typename bto_contract2_clst_base<N, M, K, T>::contr_list contr_list;

private:
    contraction2<N, M, K> m_contr; //!< Contraction descriptor
    block_tensor_i<N + K, T> &m_bta; //!< First block tensor (A)
    block_tensor_i<M + K, T> &m_btb; //!< Second block tensor (B)
    const orbit_list<N + K, T> &m_ola; //!< List of orbits in A
    const orbit_list<M + K, T> &m_olb; //!< List of orbits in B
    dimensions<N + K> m_bidimsa; //!< Block index dimensions (A)
    dimensions<M + K> m_bidimsb; //!< Block index dimensions (B)
    dimensions<N + M> m_bidimsc; //!< Block index dimensions (C)
    index<N + M> m_ic; //!< Index in C

public:
    bto_contract2_clst(
        const contraction2<N, M, K> &contr,
        block_tensor_i<N + K, T> &bta,
        block_tensor_i<M + K, T> &btb,
        const orbit_list<N + K, T> &ola,
        const orbit_list<M + K, T> &olb,
        const dimensions<N + K> &bidimsa,
        const dimensions<M + K> &bidimsb,
        const dimensions<N + M> &bidimsc,
        const index<N + M> &ic);

    void build_list(bool testzero);

private:
    /** \brief Private copy constructor
     **/
    bto_contract2_clst(const bto_contract2_clst&);

};


/** \brief Computes the list of block contractions required to compute
        a block in C (specialized for direct product, K = 0)

    \sa bto_contract2

    \ingroup ccman2_block_tensor_btod
 **/
template<size_t N, size_t M, typename T>
class bto_contract2_clst<N, M, 0, T> :
    public bto_contract2_clst_base<N, M, 0, T> {

public:
    static const char *k_clazz; //!< Class name

public:
    typedef typename bto_contract2_clst_base<N, M, 0, T>::contr_pair contr_pair;
    typedef typename bto_contract2_clst_base<N, M, 0, T>::contr_list contr_list;

private:
    contraction2<N, M, 0> m_contr; //!< Contraction descriptor
    block_tensor_i<N, T> &m_bta; //!< First block tensor (A)
    block_tensor_i<M, T> &m_btb; //!< Second block tensor (B)
    const orbit_list<N, T> &m_ola; //!< List of orbits in A
    const orbit_list<M, T> &m_olb; //!< List of orbits in B
    dimensions<N> m_bidimsa; //!< Block index dimensions (A)
    dimensions<M> m_bidimsb; //!< Block index dimensions (B)
    dimensions<N + M> m_bidimsc; //!< Block index dimensions (C)
    index<N + M> m_ic; //!< Index in C

public:
    bto_contract2_clst(
        const contraction2<N, M, 0> &contr,
        block_tensor_i<N, T> &bta,
        block_tensor_i<M, T> &btb,
        const orbit_list<N, T> &ola,
        const orbit_list<M, T> &olb,
        const dimensions<N> &bidimsa,
        const dimensions<M> &bidimsb,
        const dimensions<N + M> &bidimsc,
        const index<N + M> &ic);

    void build_list(bool testzero);

private:
    /** \brief Private copy constructor
     **/
    bto_contract2_clst(const bto_contract2_clst&);

};


} // namespace libtensor

#endif // LIBTENSOR_BTO_CONTRACT2_CLST_H
