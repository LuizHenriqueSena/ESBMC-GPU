#ifndef __CIVL_METRICS_COLLECTOR_T_H
#define __CIVL_METRICS_COLLECTOR_T_H

#include <boost/chrono.hpp>
#include <string>
#include <vector>
#include "metrics_collector_t.h"

#define CIVL_COMMAND "/usr/bin/time -v timeout 600s civl verify main.cu 2>&1"
#define PROPERTY_HOLDS "The standard properties hold for all executions."
#define REMOVE_TEMP_CIVL_FILES_COMMAND "rm -rf CIVILREP/"
#define RESULT_DESCRIPTION "=== Result ==="

class civl_metrics_collector_t : public metrics_collector_t
{
public:
    civl_metrics_collector_t()
    {
    }

    ~civl_metrics_collector_t()
    {
    }

    void prepare_verification_task()
    {
        m_command = CIVL_COMMAND;
    }

    std::string run_verification_task()
    {
        bool successful = false, supported = false;
        double system_cpu_time, user_cpu_time;
        double system_cpu_time_reference, user_cpu_time_reference;
        double current_cpu_time, current_wall_time;
        double current_cpu_time_reference, current_wall_time_reference;
        boost::chrono::process_real_cpu_clock::time_point w_start = boost::chrono::process_real_cpu_clock::now();
        boost::chrono::process_system_cpu_clock::time_point sc_start = boost::chrono::process_system_cpu_clock::now();
        boost::chrono::process_user_cpu_clock::time_point uc_start = boost::chrono::process_user_cpu_clock::now();
        std::vector<std::string> run_command_output = utils::execute_command(m_command.c_str());
        boost::chrono::process_real_cpu_clock::time_point w_end = boost::chrono::process_real_cpu_clock::now();
        boost::chrono::process_system_cpu_clock::time_point sc_end = boost::chrono::process_system_cpu_clock::now();
        boost::chrono::process_user_cpu_clock::time_point uc_end = boost::chrono::process_user_cpu_clock::now();
        current_wall_time = boost::chrono::duration_cast<boost::chrono::nanoseconds>(w_end - w_start).count() / BILLION;
        system_cpu_time = boost::chrono::duration_cast<boost::chrono::nanoseconds>(sc_end - sc_start).count() / BILLION;
        user_cpu_time = boost::chrono::duration_cast<boost::chrono::nanoseconds>(uc_end - uc_start).count() / BILLION;
        current_cpu_time = system_cpu_time + user_cpu_time;
        parse_time_output(run_command_output,
            system_cpu_time_reference, user_cpu_time_reference, current_wall_time_reference);
        current_cpu_time_reference = system_cpu_time_reference + user_cpu_time_reference;
        m_cpu_time += current_cpu_time;
        m_wall_time += current_wall_time;
        m_cpu_time_reference += current_cpu_time_reference;
        m_wall_time_reference += current_wall_time_reference;
        for (const auto &message : run_command_output)
        {
            if (message.find(RESULT_DESCRIPTION) != std::string::npos)
                supported = true;
            if (message.find(PROPERTY_HOLDS) != std::string::npos)
            {
                successful = true;
                break;
            }
        }
        std::string expected_result = get_expected_result();
        std::string actual_result;
        if (!supported)
        {
            ++m_not_supported;
            actual_result = NOT_SUPPORTED_RESULT;
        }
        else
        {
            if (successful && expected_result.compare(VERIFICATION_SUCCESSFUL_MESSAGE) == 0)
            {
                ++m_correct;
                actual_result = CORRECT_RESULT;
            }
            else if (successful && expected_result.compare(VERIFICATION_FAILED_MESSAGE) == 0)
            {
                ++m_false_correct;
                actual_result = FALSE_CORRECT_RESULT;
            }
            else if (!successful && expected_result.compare(VERIFICATION_FAILED_MESSAGE) == 0)
            {
                ++m_incorrect;
                actual_result = INCORRECT_RESULT;
            }
            else if (!successful &&
                        expected_result.compare(VERIFICATION_SUCCESSFUL_MESSAGE) == 0)
            {
                ++m_false_incorrect;
                actual_result = FALSE_CORRECT_RESULT;
            }
        }
        utils::execute_command(REMOVE_TEMP_CIVL_FILES_COMMAND);
        actual_result += " [ Boost = { Wall time: " +
            std::to_string(current_wall_time) + "s, CPU time: " +
            std::to_string(current_cpu_time) +  "s } ]";
        actual_result += " [ Time = { Wall time: " +
            std::to_string(current_wall_time_reference) + "s, CPU time: " +
            std::to_string(current_cpu_time_reference) +  "s } ]";
        return actual_result;
    }
private:
    std::string m_command;
};

#endif
