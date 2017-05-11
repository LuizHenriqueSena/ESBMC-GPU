#ifndef __GKLEE_METRICS_COLLECTOR_T_H
#define __GKLEE_METRICS_COLLECTOR_T_H

#include <boost/chrono.hpp>
#include <regex>
#include <string>
#include <vector>
#include "metrics_collector_t.h"

#define NOT_SUPPORTED 0
#define GKLEE_COMMAND "/usr/bin/time -v timeout 600s gklee main 2>&1"
#define GKLEE_PRE_COMPILE_COMMAND_KLEE "/usr/bin/time -v gklee-nvcc main.cu -libc=klee 2>&1"
#define GKLEE_PRE_COMPILE_COMMAND_UCLIBC "/usr/bin/time -v gklee-nvcc main.cu -libc=uclibc 2>&1"
#define GKLEE_COUNTEREXAMPLE_DESCRIPTION "[GKLEE]: Start executing a GPU kernel"
#define REMOVE_TEMP_GKLEE_FILES_COMMAND "(rm klee-last main main.cpp main.kernelSet.txt trace.log && rm -rf klee-out-*) 2>/dev/null"
#define SUCCESSFUL ".*KLEE]: Finishing the program.*"

class gklee_metrics_collector_t : public metrics_collector_t
{
public:
    gklee_metrics_collector_t()
    {
    }

    ~gklee_metrics_collector_t()
    {
    }

    void prepare_verification_task()
    {
        m_command = GKLEE_COMMAND;
    }

    std::string run_verification_task_UCLIBC(double& cpu_time, double& cpu_time_reference,
                                             double& wall_time, double& wall_time_reference)
    {
        bool successful = false, supported = false;
        double system_cpu_time, user_cpu_time;
        double system_cpu_time_reference, user_cpu_time_reference;
        double pre_compile_system_cpu_time_reference, pre_compile_user_cpu_time_reference;
        double current_cpu_time, current_wall_time;
        double current_cpu_time_reference, current_wall_time_reference, pre_compile_current_wall_time_reference;
        boost::chrono::process_real_cpu_clock::time_point w_start = boost::chrono::process_real_cpu_clock::now();
        boost::chrono::process_system_cpu_clock::time_point sc_start = boost::chrono::process_system_cpu_clock::now();
        boost::chrono::process_user_cpu_clock::time_point uc_start = boost::chrono::process_user_cpu_clock::now();
        std::vector<std::string> pre_compile_command_output = utils::execute_command(GKLEE_PRE_COMPILE_COMMAND_UCLIBC);
        std::vector<std::string> run_command_output = utils::execute_command(m_command.c_str());
        boost::chrono::process_real_cpu_clock::time_point w_end = boost::chrono::process_real_cpu_clock::now();
        boost::chrono::process_system_cpu_clock::time_point sc_end = boost::chrono::process_system_cpu_clock::now();
        boost::chrono::process_user_cpu_clock::time_point uc_end = boost::chrono::process_user_cpu_clock::now();
        current_wall_time = boost::chrono::duration_cast<boost::chrono::nanoseconds>(w_end - w_start).count() / BILLION;
        wall_time = current_wall_time;
        system_cpu_time = boost::chrono::duration_cast<boost::chrono::nanoseconds>(sc_end - sc_start).count() / BILLION;
        user_cpu_time = boost::chrono::duration_cast<boost::chrono::nanoseconds>(uc_end - uc_start).count() / BILLION;
        current_cpu_time = system_cpu_time + user_cpu_time;
        cpu_time = current_cpu_time;
        parse_time_output(pre_compile_command_output,
            pre_compile_system_cpu_time_reference, pre_compile_user_cpu_time_reference,
            pre_compile_current_wall_time_reference);
        parse_time_output(run_command_output,
            system_cpu_time_reference, user_cpu_time_reference, current_wall_time_reference);
        current_cpu_time_reference = system_cpu_time_reference + user_cpu_time_reference +
            pre_compile_system_cpu_time_reference + pre_compile_user_cpu_time_reference;
        current_wall_time_reference += pre_compile_current_wall_time_reference;
        cpu_time_reference = current_cpu_time_reference;
        wall_time_reference = current_wall_time_reference;
        for (const auto &message : run_command_output)
        {
            if (message.find(GKLEE_COUNTEREXAMPLE_DESCRIPTION) != std::string::npos)
                supported = true;

	        if (std::regex_match(message, std::regex(SUCCESSFUL)))
            {
                successful = true;
                break;
            }
        }
        std::string expected_result = get_expected_result();
        std::string actual_result;
        if (!supported)
        {
            actual_result = NOT_SUPPORTED_RESULT;
        }
        else
        {
            std::string expected_result = get_expected_result();
            if (successful && expected_result.compare(VERIFICATION_SUCCESSFUL_MESSAGE) == 0)
            {
                actual_result = CORRECT_RESULT;
            }
            else if (successful && expected_result.compare(VERIFICATION_FAILED_MESSAGE) == 0)
            {
                actual_result = FALSE_CORRECT_RESULT;
            }
            else if (!successful && expected_result.compare(VERIFICATION_FAILED_MESSAGE) == 0)
            {
                actual_result = INCORRECT_RESULT;
            }
            else if (!successful &&
                        expected_result.compare(VERIFICATION_SUCCESSFUL_MESSAGE) == 0)
            {
                actual_result = FALSE_INCORRECT_RESULT;
            }
        }
        utils::execute_command(REMOVE_TEMP_GKLEE_FILES_COMMAND);
        actual_result += " [ Boost = { Wall time: " +
            std::to_string(current_wall_time) + "s, CPU time: " +
            std::to_string(current_cpu_time) +  "s } ]";
        actual_result += " [ Time = { Wall time: " +
            std::to_string(current_wall_time_reference) + "s, CPU time: " +
            std::to_string(current_cpu_time_reference) +  "s } ]";
        return actual_result;
    }

    std::string run_verification_task_KLEE(double& cpu_time, double& cpu_time_reference,
                                           double& wall_time, double& wall_time_reference)
    {
        bool successful = false, supported = false;
        double system_cpu_time, user_cpu_time;
        double system_cpu_time_reference, user_cpu_time_reference;
        double pre_compile_system_cpu_time_reference, pre_compile_user_cpu_time_reference;
        double current_cpu_time, current_wall_time;
        double current_cpu_time_reference, current_wall_time_reference, pre_compile_current_wall_time_reference;
        boost::chrono::process_real_cpu_clock::time_point w_start = boost::chrono::process_real_cpu_clock::now();
        boost::chrono::process_system_cpu_clock::time_point sc_start = boost::chrono::process_system_cpu_clock::now();
        boost::chrono::process_user_cpu_clock::time_point uc_start = boost::chrono::process_user_cpu_clock::now();
        std::vector<std::string> pre_compile_command_output = utils::execute_command(GKLEE_PRE_COMPILE_COMMAND_KLEE);
        std::vector<std::string> run_command_output = utils::execute_command(m_command.c_str());
        boost::chrono::process_real_cpu_clock::time_point w_end = boost::chrono::process_real_cpu_clock::now();
        boost::chrono::process_system_cpu_clock::time_point sc_end = boost::chrono::process_system_cpu_clock::now();
        boost::chrono::process_user_cpu_clock::time_point uc_end = boost::chrono::process_user_cpu_clock::now();
        current_wall_time = boost::chrono::duration_cast<boost::chrono::nanoseconds>(w_end - w_start).count() / BILLION;
        wall_time = current_wall_time;
        system_cpu_time = boost::chrono::duration_cast<boost::chrono::nanoseconds>(sc_end - sc_start).count() / BILLION;
        user_cpu_time = boost::chrono::duration_cast<boost::chrono::nanoseconds>(uc_end - uc_start).count() / BILLION;
        current_cpu_time = system_cpu_time + user_cpu_time;
        cpu_time = current_cpu_time;
        parse_time_output(pre_compile_command_output,
            pre_compile_system_cpu_time_reference, pre_compile_user_cpu_time_reference,
            pre_compile_current_wall_time_reference);
        parse_time_output(run_command_output,
            system_cpu_time_reference, user_cpu_time_reference, current_wall_time_reference);
        current_cpu_time_reference = system_cpu_time_reference + user_cpu_time_reference +
            pre_compile_system_cpu_time_reference + pre_compile_user_cpu_time_reference;
        current_wall_time_reference += pre_compile_current_wall_time_reference;
        cpu_time_reference = current_cpu_time_reference;
        wall_time_reference = current_wall_time_reference;
        for (const auto &message : run_command_output)
        {
            if (message.find(GKLEE_COUNTEREXAMPLE_DESCRIPTION) != std::string::npos)
                supported = true;

	        if (std::regex_match(message, std::regex(SUCCESSFUL)))
            {
                successful = true;
                break;
            }
        }
        std::string expected_result = get_expected_result();
        std::string actual_result;
        if (!supported)
        {
            actual_result = NOT_SUPPORTED_RESULT;
        }
        else
        {
            std::string expected_result = get_expected_result();
            if (successful && expected_result.compare(VERIFICATION_SUCCESSFUL_MESSAGE) == 0)
            {
                actual_result = CORRECT_RESULT;
            }
            else if (successful && expected_result.compare(VERIFICATION_FAILED_MESSAGE) == 0)
            {
                actual_result = FALSE_CORRECT_RESULT;
            }
            else if (!successful && expected_result.compare(VERIFICATION_FAILED_MESSAGE) == 0)
            {
                actual_result = INCORRECT_RESULT;
            }
            else if (!successful &&
                        expected_result.compare(VERIFICATION_SUCCESSFUL_MESSAGE) == 0)
            {
                actual_result = FALSE_INCORRECT_RESULT;
            }
        }
        actual_result += " [ Boost = { Wall time: " +
            std::to_string(current_wall_time) + "s, CPU time: " +
            std::to_string(current_cpu_time) +  "s } ]";
        actual_result += " [ Time = { Wall time: " +
            std::to_string(current_wall_time_reference) + "s, CPU time: " +
            std::to_string(current_cpu_time_reference) +  "s } ]";
        return actual_result;
    }

    std::string run_verification_task()
    {
        std::string actual_result_UCLIBC;
        std::string actual_result_KLEE;
        double cpu_time_UCLIBC = 0.0, wall_time_UCLIBC = 0.0;
        double cpu_time_reference_UCLIBC = 0.0, wall_time_reference_UCLIBC = 0.0;
        double cpu_time_KLEE = 0.0, wall_time_KLEE = 0.0;
        double cpu_time_reference_KLEE = 0.0, wall_time_reference_KLEE = 0.0;
        actual_result_UCLIBC = run_verification_task_UCLIBC(cpu_time_UCLIBC, cpu_time_reference_UCLIBC,
                                                            wall_time_UCLIBC, wall_time_reference_UCLIBC);
        actual_result_KLEE = run_verification_task_KLEE(cpu_time_KLEE, wall_time_KLEE,
                                                        cpu_time_reference_KLEE, wall_time_reference_KLEE);

        if (actual_result_UCLIBC.find(CORRECT_RESULT) != std::string::npos ||
            actual_result_KLEE.find(CORRECT_RESULT) != std::string::npos)
        {
            ++m_correct;
            if (actual_result_UCLIBC.find(CORRECT_RESULT) != std::string::npos &&
                actual_result_KLEE.find(CORRECT_RESULT) != std::string::npos)
            {
                if(wall_time_UCLIBC < wall_time_KLEE)
                {
                    m_cpu_time += cpu_time_UCLIBC;
                    m_wall_time += wall_time_UCLIBC;
                    m_cpu_time_reference += cpu_time_reference_UCLIBC;
                    m_wall_time_reference += wall_time_reference_UCLIBC;
                    return actual_result_UCLIBC;
                }
                else
                {
                    m_cpu_time += cpu_time_KLEE;
                    m_wall_time += wall_time_KLEE;
                    m_cpu_time_reference += cpu_time_reference_KLEE;
                    m_wall_time_reference += wall_time_reference_KLEE;
                    return actual_result_KLEE;
                }
            }
            else if (actual_result_UCLIBC.find(CORRECT_RESULT) != std::string::npos)
            {
                m_cpu_time += cpu_time_UCLIBC;
                m_wall_time += wall_time_UCLIBC;
                m_cpu_time_reference += cpu_time_reference_UCLIBC;
                m_wall_time_reference += wall_time_reference_UCLIBC;
                return actual_result_UCLIBC;
            }
            else
            {
                m_cpu_time += cpu_time_KLEE;
                m_wall_time += wall_time_KLEE;
                m_cpu_time_reference += cpu_time_reference_KLEE;
                m_wall_time_reference += wall_time_reference_KLEE;
                return actual_result_KLEE;
            }

        }
        else if (actual_result_UCLIBC.find(NOT_SUPPORTED_RESULT) != std::string::npos ||
                 actual_result_KLEE.find(NOT_SUPPORTED_RESULT) != std::string::npos)
        {
            if (actual_result_UCLIBC.find(NOT_SUPPORTED_RESULT) != std::string::npos &&
                actual_result_KLEE.find(NOT_SUPPORTED_RESULT) != std::string::npos)
            {
                ++m_not_supported;
                if(wall_time_UCLIBC < wall_time_KLEE)
                {
                    m_cpu_time += cpu_time_UCLIBC;
                    m_wall_time += wall_time_UCLIBC;
                    m_cpu_time_reference += cpu_time_reference_UCLIBC;
                    m_wall_time_reference += wall_time_reference_UCLIBC;
                    return actual_result_UCLIBC;
                }
                else
                {
                    m_cpu_time += cpu_time_KLEE;
                    m_wall_time += wall_time_KLEE;
                    m_cpu_time_reference += cpu_time_reference_KLEE;
                    m_wall_time_reference += wall_time_reference_KLEE;
                    return actual_result_KLEE;
                }
            }
            else if (actual_result_UCLIBC.find(NOT_SUPPORTED_RESULT) != std::string::npos)
            {
                m_cpu_time += cpu_time_KLEE;
                m_wall_time += wall_time_KLEE;
                m_cpu_time_reference += cpu_time_reference_KLEE;
                m_wall_time_reference += wall_time_reference_KLEE;
                return actual_result_KLEE;
            }
            else
            {
                m_cpu_time += cpu_time_UCLIBC;
                m_wall_time += wall_time_UCLIBC;
                m_cpu_time_reference += cpu_time_reference_UCLIBC;
                m_wall_time_reference += wall_time_reference_UCLIBC;
                return actual_result_UCLIBC;
            }
        }
        else
        {
            if(wall_time_UCLIBC < wall_time_KLEE)
            {
                if (actual_result_UCLIBC.find(INCORRECT_RESULT) != std::string::npos)
                {
                    ++m_incorrect;
                }
                else if (actual_result_UCLIBC.find(FALSE_CORRECT_RESULT) != std::string::npos)
                {
                    ++m_false_correct;
                }
                else
                {
                    ++m_false_incorrect; // FALSE_INCORRECT_RESULT
                }
                m_cpu_time += cpu_time_UCLIBC;
                m_wall_time += wall_time_UCLIBC;
                m_cpu_time_reference += cpu_time_reference_UCLIBC;
                m_wall_time_reference += wall_time_reference_UCLIBC;
                return actual_result_UCLIBC;
            }
            else
            {
                if(actual_result_KLEE.find(INCORRECT_RESULT) != std::string::npos)
                {
                    ++m_incorrect;
                }
                else if (actual_result_KLEE.find(FALSE_CORRECT_RESULT) != std::string::npos)
                {
                    ++m_false_correct;
                }
                else
                {
                    ++m_false_incorrect; // FALSE_INCORRECT_RESULT
                }
                m_cpu_time += cpu_time_KLEE;
                m_wall_time += wall_time_KLEE;
                m_cpu_time_reference += cpu_time_reference_KLEE;
                m_wall_time_reference += wall_time_reference_KLEE;
                return actual_result_KLEE;
            }
        }
    }
private:
    std::string m_command;
    unsigned int m_compiler_to_use;
};

#endif
