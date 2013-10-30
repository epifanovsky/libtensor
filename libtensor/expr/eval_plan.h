#ifndef LIBTENSOR_EXPR_EVAL_PLAN_H
#define LIBTENSOR_EXPR_EVAL_PLAN_H

#include <vector>
#include "node_assign.h"

namespace libtensor {
namespace expr {


/** \brief Action codes in the evaluation plan

    \ingroup libtensor_expr
 **/
struct eval_plan_action_code {
    enum ac {
        ASSIGN, //!< Assign to a variable
        CREATE_INTERM, //!< Create an intermediate
        DELETE_INTERM //!< Delete an intermediate
    };
};


/** \brief Item of the evaluation plan

    Contains one instruction of the evaluation plan. The following three actions
    are possible:
     - create an intermediate,
     - delete an intermediate,
     - perform an assignment.

    \sa eval_plan, eval_plan_action_code, node_assign

    \ingroup libtensor_expr
 **/
struct eval_plan_item {
public:
    typedef node::tid_t tid_t; //!< Tensor ID type

public:
    eval_plan_action_code::ac code; //!< Action code
    tid_t tid; //!< Intermediate tensor id
    const node_assign *node; //!< Assignment

public:
    eval_plan_item(eval_plan_action_code::ac code_, tid_t tid_) :
        code(code_), tid(tid_), node(0)
    { }

    eval_plan_item(eval_plan_action_code::ac code_, const node_assign *node_) :
        code(code_), tid(0), node(node_)
    { }

};


/** \brief Evaluation plan of an expression

    Contains a list of instructions for evaluating an expression. Each node
    represents one assignment instruction.

    \sa node_assign

    \ingroup libtensor_expr
 **/
class eval_plan {
private:
    std::vector<eval_plan_item> m_lst; //!< List of nodes

public:
    typedef eval_plan_item::tid_t tid_t; //!< Tensor ID type
    typedef std::vector<eval_plan_item>::const_iterator iterator; //!< Iterator

public:
    //! \name Construction and destruction
    //!{

    /** \brief Destroys the object
     **/
    ~eval_plan() {
        for(size_t i = 0; i < m_lst.size(); i++) delete m_lst[i].node;
    }

    //!}

    //! \name Iteration over the list of items
    //!{

    /** \brief Returns the iterator to the first node
     **/
    iterator begin() const {
        return m_lst.begin();
    }

    /** \brief Returns the iterator to after the last node
     **/
    iterator end() const {
        return m_lst.end();
    }

    /** \brief Returns the action item pointed by the given iterator
     **/
    const eval_plan_item &get_item(const iterator &i) const {
        return *i;
    }

    //!}

    //! \name Building of the list of items

    //!{

    /** \brief Adds an intermediate creation instruction
     **/
    void create_intermediate(tid_t tid) {
        m_lst.push_back(eval_plan_item(
            eval_plan_action_code::CREATE_INTERM, tid));
    }

    /** \brief Adds an intermediate deletion instruction
     **/
    void delete_intermediate(tid_t tid) {
        m_lst.push_back(eval_plan_item(
            eval_plan_action_code::DELETE_INTERM, tid));
    }

    /** \brief Adds an assignment node to the end of the list
     **/
    void insert_assignment(const node_assign &n) {
        m_lst.push_back(eval_plan_item(
            eval_plan_action_code::ASSIGN, new node_assign(n)));
    }

    //!}

};


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_EVAL_PLAN_H
