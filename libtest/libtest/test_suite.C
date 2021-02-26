#include <cstdio>
#include <exception>
#include "test_suite.h"

namespace libtest {


test_suite::test_suite(const char *name) : m_name(name) { 

    m_handler = 0;
}


test_suite::~test_suite() {

}


void test_suite::add_test(const char *name, unit_test_factory_i &utf) {

    m_tests[name] = &utf;
}


unsigned test_suite::get_num_tests() const {

    return m_tests.size();
}


unsigned test_suite::run_test(const char *name) {

    evt_suite_start();
    std::map<std::string,unit_test_factory_i*>::iterator i = m_tests.find(name);
    if(i == m_tests.end()) return 0;
    unit_test *t = i->second->create_instance();
    evt_test_start(name);
    try {
        t->perform();
        delete t; t = 0;
        evt_test_end_success(name);
    } catch(test_exception e) {
        delete t; t = 0;
        evt_test_end_exception(name, e);
    } catch(std::exception e) {
        delete t; t = 0;
        char s[1024];
        snprintf(s, 1024, "std::exception:\n%s", e.what());
        test_exception texc("libtest::test_suite::run_test"
            "(const char*)", __FILE__, __LINE__, s);
        evt_test_end_exception(name, texc);
    } catch(...) {
        delete t; t = 0;
        test_exception texc("libtest::test_suite::run_test"
            "(const char*)", __FILE__, __LINE__,
            "Unknown exception");
        evt_test_end_exception(name, texc);
    }
    evt_suite_end();
    return 1;
}


unsigned test_suite::run_all_tests() {

    unsigned n_tests = 0;
    evt_suite_start();
    std::map<std::string,unit_test_factory_i*>::iterator i = m_tests.begin();
    while(i != m_tests.end()) {
        unit_test *t = i->second->create_instance();
        evt_test_start(i->first.c_str());
        try {
            t->perform();
            delete t; t = 0;
            evt_test_end_success(i->first.c_str());
        } catch(test_exception e) {
            delete t; t = 0;
            evt_test_end_exception(i->first.c_str(), e);
        }
        n_tests++;
        i++;
    }
    evt_suite_end();
    return n_tests;
}


void test_suite::set_handler(suite_event_handler *handler) {

    m_handler = handler;
}


void test_suite::evt_suite_start() {

    if(m_handler) m_handler->on_suite_start(m_name.c_str());
}


void test_suite::evt_suite_end() {

    if(m_handler) m_handler->on_suite_end(m_name.c_str());
}


void test_suite::evt_test_start(const char *test) {

    if(m_handler) m_handler->on_test_start(test);
}


void test_suite::evt_test_end_success(const char *test) {

    if(m_handler) m_handler->on_test_end_success(test);
}


void test_suite::evt_test_end_exception(const char *test,
    const test_exception &e) {

    if(m_handler) m_handler->on_test_end_exception(test, e);
}


} // namespace libtest

