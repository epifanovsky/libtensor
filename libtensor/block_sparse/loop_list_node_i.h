#ifndef LOOP_LIST_NODE_I_H
#define LOOP_LIST_NODE_I_H

/* General interface for sparse and dense loop list nodes
 *
 */
class loop_list_node_i
{

    /** \brief Returns the upper bound of the loop
     **/
    virtual size_t weight() const = 0;

    /** \brief Returns increment of array i from iteration   
     **/
    virtual size_t stepa(size_t i) = 0;
    virtual size_t stepb(size_t i) = 0;
}

#endif /* LOOP_LIST_I_H */
