#ifndef __METRICS_COLLECTOR_T_H
#define __METRICS_COLLECTOR_T_H

#include <fstream>
#include <iostream>
#include <regex>
#include <string>
#include <vector>
#include "utils.h"

#define BILLION 1E9
#define MINUTES_TO_SECONDS_FACTOR 60
#define CORRECT_RESULT "[CORRECT]"
#define FALSE_CORRECT_RESULT "[FALSE CORRECT]"
#define FALSE_INCORRECT_RESULT "[FALSE INCORRECT]"
#define INCORRECT_RESULT "[INCORRECT]"
#define NOT_SUPPORTED_RESULT "[NOT SUPPORTED]"
#define SEMICOLON ":"
#define SYSTEM_CPU_TIME_REGEX ".*System time \\(seconds\\):.*"
#define TEST_DESCRIPTION_DELIMITER '^'
#define TEST_DESCRIPTION_FILE "test.desc"
#define USER_CPU_TIME_REGEX ".*User time \\(seconds\\):.*"
#define VERIFICATION_FAILED_MESSAGE "VERIFICATION FAILED"
#define VERIFICATION_SUCCESSFUL_MESSAGE "VERIFICATION SUCCESSFUL"
#define WALL_TIME_REGEX ".*Elapsed \\(wall clock\\) time \\(h:mm:ss or m:ss\\):.*"
#define WHITESPACE " "

class metrics_collector_t {
public:
    metrics_collector_t() : m_correct(0), m_false_correct(0),
                            m_false_incorrect(0), m_incorrect(0), m_not_supported(0),
                            m_cpu_time(0.0), m_cpu_time_reference(0.0),
                            m_wall_time(0.0), m_wall_time_reference(0.0)
    {
    }

    virtual ~metrics_collector_t() = 0;

    void parse_time_output(const std::vector<std::string>& output,
                           double& system_cpu_time,
                           double& user_cpu_time,
                           double& wall_time) const;

    virtual void prepare_verification_task() = 0;

    std::string get_expected_result() const;

    virtual std::string run_verification_task() = 0;

    unsigned int correct() const
    {
        return m_correct;
    }

    unsigned int false_correct() const
    {
        return m_false_correct;
    }

    unsigned int false_incorrect() const
    {
        return m_false_incorrect;
    }

    unsigned int incorrect() const
    {
        return m_incorrect;
    }

    unsigned int not_supported() const
    {
        return m_not_supported;
    }

    double cpu_time() const
    {
        return m_cpu_time;
    }

    double cpu_time_reference() const
    {
        return m_cpu_time_reference;
    }

    double wall_time() const
    {
        return m_wall_time;
    }

    double wall_time_reference() const
    {
        return m_wall_time_reference;
    }
protected:
    unsigned int m_correct;
    unsigned int m_false_correct;
    unsigned int m_false_incorrect;
    unsigned int m_incorrect;
    unsigned int m_not_supported;
    double m_cpu_time;
    double m_cpu_time_reference;
    double m_wall_time;
    double m_wall_time_reference;
};

#endif
