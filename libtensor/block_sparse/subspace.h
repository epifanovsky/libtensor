#ifndef SUBSPACE_H
#define SUBSPACE_H

#include "sparse_defs.h"
#include "../core/out_of_bounds.h"


namespace libtensor {

class subspace 
{
private:
    size_t m_dim;
    idx_list m_abs_indices;
public:
    static const char* k_clazz; //!< Class name

    /** \brief Creates the subspace with a given dimension
        \param dim Number of elements in this space.
     **/
    explicit subspace(size_t dim,
                      const idx_list& split_points = idx_list(1,0));
    
    /** \brief Returns the dimension of the block index space 
     **/
    size_t get_dim() const;

    size_t get_nnz() const { return m_dim; }

    /** \brief Returns the number blocks into which this space has been split 
     **/
    size_t get_n_blocks() const;

    /** \brief Splits this space into blocks with offsets starting at offsets
               in split_points. First block always starts at zero
        \param split_points Iterable container of absolute indices where each block should start 
               If split_points begins with zero, that value is ignored
        \throw out_of_bounds If a split_point value exceeds the index limits, or if a zero length vector is passed 
     **/
    void split(const std::vector<size_t>& split_points)
        throw(out_of_bounds);

    /** \brief Returns the size of the block with block index block_idx
        \throw out_of_bounds If (# of blocks  - 1) < block_idx || block_idx < 0
     **/
    size_t get_block_size(size_t block_idx) const 
        throw(out_of_bounds);

    /** \brief Returns the absolute starting index of the block with block index block_idx
        \throw out_of_bounds If (# of blocks  - 1) < block_idx < 0
     **/
    size_t get_block_abs_index(size_t block_idx) const 
        throw(out_of_bounds);

    /** \brief Returns whether this object is equal to another. 
     *         Equality is defined to be the same dimension and block splitting pattern
     **/
    bool operator==(const subspace& rhs) const;

    /** \brief Returns whether this object is not equal to another. 
     *         Equality is defined to be the same dimension and block splitting pattern
     **/
    bool operator!=(const subspace& rhs) const;
};

} // namespace libtensor

#endif /* SUBSPACE_H */
