#ifndef LIBTENSOR_SE_PART_H
#define LIBTENSOR_SE_PART_H

#include <libtensor/core/symmetry_element_i.h>

namespace libtensor {


/** \brief Symmetry between %tensor partitions
    \tparam N Symmetry cardinality (%tensor order).
    \tparam T Tensor element type.

    This %symmetry element establishes relationships between partitions
    of a block %tensor. Each partition consists of one or more adjacent
    blocks.

    Tensor indexes that are affected by this %symmetry element are
    specified using a mask.

    The number of partitions specifies how blocks will be grouped together.
    For the block %index space to be valid with the %symmetry element,
    the number of blocks along each affected dimension must be divisible
    by the number of partitions. Moreover, block sizes must correspond
    correctly from partition to partition. That is, if the partitions must
    have the same block structure.

    TODO: 
    - separate class definition and implementation
    - replace sign by scalar_transf

    \ingroup libtensor_symmetry
**/
template<size_t N, typename T>
class se_part : public symmetry_element_i<N, T> {
public:
    static const char *k_clazz; //!< Class name
    static const char *k_sym_type; //!< Symmetry type

private:
    block_index_space<N> m_bis; //!< Block %index space
    dimensions<N> m_bidims; //!< Block %index space dimensions
    dimensions<N> m_pdims; //!< Partition %index dimensions
    size_t *m_fmap; //!< Forward mapping
    size_t *m_rmap; //!< Reverse mapping
    scalar_transf<T> *m_ftr; //!< Transforms of the mappings

public:
    //!	\name Construction and destruction / assignment
    //@{

    /**	\brief Initializes the %symmetry element
        \param bis Block %index space.
        \param msk Mask of affected dimensions.
        \param npart Number of partitions along each dimension.
    **/
    se_part(const block_index_space<N> &bis, const mask<N> &msk, size_t npart);

    /** \brief Initializes the %symmetry element (varying number of partitions)
        \param bis Block %index space.
        \param pdims Partition dimensions.
    **/
    se_part(const block_index_space<N> &bis, const dimensions<N> &pdims);

    /**	\brief Copy constructor
     **/
    se_part(const se_part<N, T> &elem);

    /**	\brief Virtual destructor
     **/
    virtual ~se_part();

    //@}

    //!	\name Manipulations
    //@{

    /**	\brief Adds a mapping between two partitions
        \param idx1 First partition %index.
        \param idx2 Second partition %index.
        \param sign Sign of the mapping (true positive, false negative)
    **/
    void add_map(const index<N> &idx1, const index<N> &idx2,
            const scalar_transf<T> &tr = scalar_transf<T>());

    /** \brief Marks a partition as not allowed (i.e. all blocks in it
        are not allowed)

        If a mapping exist that includes the partition, all partitions in the
        mapping are marked as forbidden.

        \param idx Partition %index.
    **/
    void mark_forbidden(const index<N> &idx);

    //@}

    //! \name Access functions

    //@{

    /** \brief Returns the block index space for the partitions.
     **/
    const block_index_space<N> &get_bis() const {
        return m_bis;
    }

    /** \brief Returns the partition dimensions.
     **/
    const dimensions<N> &get_pdims() const {
        return m_pdims;
    }

    /** \brief Checks if the partition is forbidden
        \param idx Partition %index

        This function yields similar functionality as is_allowed(), but it
        answers the negative question for partitions instead of blocks.
    **/
    bool is_forbidden(const index<N> &idx) const;

    /** \brief Returns the index to which idx is mapped directly
        (refers to forward mapping)
        \param idx Start index of map.
        \return End index of the direct map.
    **/
    index<N> get_direct_map(const index<N> &idx) const;

    /** \brief Returns the sign of the map between the two indexes.
        \param from First index.
        \param to Second index.
        \return True for even map, false for odd (-1) map.
    **/
    scalar_transf<T> get_transf(const index<N> &from, const index<N> &to) const;

    /** \brief Check if there exists a map between two indexes
        \param from First index.
        \param to Second index.
        \return True, if map exists.
    **/
    bool map_exists(const index<N> &from, const index<N> &to) const;

    /** \brief Permute the dimensions of the symmetry element
        \param perm Permutation
    **/
    void permute(const permutation<N> &perm);

    //@}

    //!	\name Implementation of symmetry_element_i<N, T>
    //@{

    /**	\copydoc symmetry_element_i<N, T>::get_type()
     **/
    virtual const char *get_type() const {
        return k_sym_type;
    }

    /**	\copydoc symmetry_element_i<N, T>::clone()
     **/
    virtual symmetry_element_i<N, T> *clone() const {
        return new se_part<N, T>(*this);
    }

    /**	\copydoc symmetry_element_i<N, T>::is_valid_bis
     **/
    virtual bool is_valid_bis(const block_index_space<N> &bis) const;

    /**	\copydoc symmetry_element_i<N, T>::is_allowed
     **/
    virtual bool is_allowed(const index<N> &idx) const;

    /**	\copydoc symmetry_element_i<N, T>::apply(index<N>&)
     **/
    virtual void apply(index<N> &idx) const;

    /**	\copydoc symmetry_element_i<N, T>::apply(
            index<N>&, tensor_transf<N, T>&)
    **/
    virtual void apply(index<N> &idx, tensor_transf<N, T> &tr) const;

    //@}

private:
    /**	\brief Builds the partition %dimensions, throws an exception
        if the arguments are invalid
    **/
    static dimensions<N> make_pdims(const block_index_space<N> &bis,
            const mask<N> &msk, size_t npart);

    /** \brief Returns true if the partition %dimensions are valid in the
        block index space
    **/
    static bool is_valid_pdims(const block_index_space<N> &bis,
            const dimensions<N> &d);

    /** Adds the map a->b to the loop a is in.
     **/
    void add_to_loop(size_t a, size_t b, const scalar_transf<T> &tr);

    /**	\brief Returns true if the %index is a valid partition %index,
        false otherwise
    **/
    bool is_valid_pidx(const index<N> &idx);

};

} // namespace libtensor

#endif // LIBTENSOR_SE_PART_H

