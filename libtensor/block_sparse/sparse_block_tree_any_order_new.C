#include "sparse_block_tree_any_order_new.h"
#include "sparse_block_tree_iterator_new.h"

namespace libtensor {

namespace impl {

bool sparse_block_tree_any_order_new::operator==(const sparse_block_tree_any_order_new& rhs) const
{
    for(size_t i = 0; i < m_keys.size(); ++i)
    {
        if(m_keys[i] != rhs.m_keys[i])
        {
            return false;
        }
        if(m_order == 1)
        {
            if(m_values[i] != rhs.m_values[i])
            {
                return false;
            }
        }
        else
        {
            if((*m_children[i]) != (*rhs.m_children[i]))
            {
                return false;
            }
        }
    }
    return true;
}

bool sparse_block_tree_any_order_new::operator!=(const sparse_block_tree_any_order_new& rhs) const
{
    return !(*this == rhs);
}

sparse_block_tree_iterator_new<false> sparse_block_tree_any_order_new::begin()
{
    if(m_keys.size() == 0)
    {
        return iterator(NULL);
    }
    else
    {
        return iterator(this);
    }
}

sparse_block_tree_iterator_new<false> sparse_block_tree_any_order_new::end()
{
    return iterator(NULL);
}

} // namespace impl

} // namespace libtensor
