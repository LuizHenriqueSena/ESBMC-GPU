#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <regex>
#include <string>
#include <vector>
#include <cstdio>
#include <dirent.h>
#include <unistd.h>

using namespace std;

#define BILLION 1E9
#define BUFFER_SIZE 128
#define NOT_SUPPORTED 0
#define USE_KLEE 1
#define USE_UCLIBC 2
#define CURRENT_FOLDER_SYMLINK "."
#define ERRORS_GENERATED_REGEX ".*\\d+ error[s]* generated."
#define GKLEE_PRE_COMPILE_COMMAND_KLEE "gklee-nvcc main.cu -libc=klee"
#define GKLEE_PRE_COMPILE_COMMAND_UCLIBC "gklee-nvcc main.cu -libc=uclibc"
#define GKLEE_RUN_COMMAND "gklee main"
#define KLEE_ERROR_REGEX "KLEE: ERROR:"
#define PARENT_FOLDER_SYMLINK ".."
#define PATH_SEPARATOR "/"
#define PIPE_READ_MODE "r"
#define SEGMENTATION_FAULT_REGEX ".*Segmentation fault      \\(core dumped\\) klee.*"
#define TEST_DESCRIPTION_DELIMITER '^'
#define TEST_DESCRIPTION_FILE "test.desc"
#define VERIFICATION_FAILED_MESSAGE "VERIFICATION FAILED"
#define VERIFICATION_SUCCESSFUL_MESSAGE "VERIFICATION SUCCESSFUL"

vector<string> execute_command(const char *command) {
    array<char, BUFFER_SIZE> buffer;
    vector<string> result;
    shared_ptr<FILE> pipe(popen(command, PIPE_READ_MODE), pclose);
    if (pipe)
    {
        while (!feof(pipe.get()))
        {
            if (fgets(buffer.data(), BUFFER_SIZE, pipe.get()) != nullptr)
            {
                auto data = string(buffer.data());
                result.push_back(data.substr(0, data.length() - 1));
            }
        }
    }
    return result;
}

vector<string> list_directories()
{
    DIR *dir;
    struct dirent *ent;
    vector<string> directories;
    if ((dir = opendir(CURRENT_FOLDER_SYMLINK)) != nullptr)
    {
        while ((ent = readdir(dir)) != nullptr)
        {
            if (ent->d_type == DT_DIR)
            {
                auto current_directory = string(ent->d_name);
                if (current_directory.compare(CURRENT_FOLDER_SYMLINK) != 0 &&
                    current_directory.compare(PARENT_FOLDER_SYMLINK) != 0)
                    directories.push_back(current_directory);
            }
        }
        closedir(dir);
        sort(directories.begin(), directories.end());
    }
    return directories;
}

string get_result()
{
    ifstream test_description;
    string line, result;
    test_description.open(TEST_DESCRIPTION_FILE);
    while(!(test_description.eof()))
    {
       getline(test_description, line);
       if (line.at(0) == TEST_DESCRIPTION_DELIMITER)
       {
          result = line.substr(1, line.size() - 2);
          break;
       }
    }
    test_description.close();
    return result;
}

unsigned int reason_about_test_case()
{
    auto pre_compile_ulibc_output = execute_command(GKLEE_PRE_COMPILE_COMMAND_UCLIBC);
    bool ulibc_supported = true, klee_supported = true;
    auto ulibc_start_wall_time = chrono::steady_clock::now();
    for (const auto &message : pre_compile_ulibc_output)
    {
        if (regex_match(message, regex(ERRORS_GENERATED_REGEX)))
        {
            ulibc_supported = false;
            break;
        }
    }
    auto ulibc_end_wall_time = chrono::steady_clock::now();
    auto klee_start_wall_time = chrono::steady_clock::now();
    auto pre_compile_klee_output = execute_command(GKLEE_PRE_COMPILE_COMMAND_KLEE);
    for (const auto &message : pre_compile_klee_output)
    {
        if (regex_match(message, regex(ERRORS_GENERATED_REGEX)))
        {
            klee_supported = false;
            break;
        }
    }
    auto klee_end_wall_time = chrono::steady_clock::now();
    double ulibc_wall_time = chrono::duration_cast<chrono::nanoseconds>(ulibc_end_wall_time - ulibc_start_wall_time).count() / BILLION;
    double klee_wall_time = chrono::duration_cast<chrono::nanoseconds>(klee_end_wall_time - klee_start_wall_time).count() / BILLION;
    if (ulibc_supported && klee_supported)
        return ulibc_wall_time < klee_wall_time ? USE_UCLIBC : USE_KLEE;
    else if (ulibc_supported)
        return USE_UCLIBC;
    else if (klee_supported)
        return USE_KLEE;
    else
        return NOT_SUPPORTED;
}

void run_gklee(unsigned int &correct, unsigned int &false_correct, unsigned int &false_incorrect,
               unsigned int &incorrect, unsigned int &not_supported, unsigned int pre_compiler)
{
    bool supported = true, successful = true;
    if (pre_compiler == USE_UCLIBC)
        execute_command(GKLEE_PRE_COMPILE_COMMAND_UCLIBC);
    else if (pre_compiler == USE_KLEE)
        execute_command(GKLEE_PRE_COMPILE_COMMAND_KLEE);
    else
        supported = false;
    if (supported)
    {
        auto run_command_output = execute_command(GKLEE_RUN_COMMAND);
        for (const auto &message : run_command_output)
        {
            if (regex_match(message, regex(SEGMENTATION_FAULT_REGEX)))
            {
                supported = false;
                break;
            }
            else if (message.find(KLEE_ERROR_REGEX) != string::npos)
            {
                successful = false;
                break;
            }
        }
    }
    if (!supported)
    {
        ++not_supported;
    }
    else
    {
        string result = get_result();
        if (successful && result.compare(VERIFICATION_SUCCESSFUL_MESSAGE) == 0)
            ++correct;
        else if (successful && result.compare(VERIFICATION_FAILED_MESSAGE) == 0)
            ++false_correct;
        else if (!successful && result.compare(VERIFICATION_FAILED_MESSAGE) == 0)
            ++incorrect;
        else if (!successful && result.compare(VERIFICATION_SUCCESSFUL_MESSAGE) == 0)
            ++false_incorrect;
    }
}

void calculate_verification_rate()
{
    unsigned int correct = 0, false_correct = 0, false_incorrect = 0, incorrect = 0, not_supported = 0;
    timespec start, end, inner_start, inner_end;
    chrono::steady_clock::time_point inner_start_steady, inner_end_steady;
    vector<string> suites;
    vector<string> sub_suites;
    vector<string> test_cases;
    suites = list_directories();
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start); // set total start time
    auto start_steady = chrono::steady_clock::now();
    double offsets_wall_time = 0.0;
    double offsets_cpu_time = 0.0;
    for (auto suite = suites.begin(); suite != suites.end(); ++suite)
    {
        chdir((*suite).c_str());
        cout << "========== SUITE: " << *suite << " ========== " << endl << endl;
        sub_suites = list_directories();
        for (auto sub_suite = sub_suites.begin(); sub_suite != sub_suites.end(); ++sub_suite)
        {
            chdir((*sub_suite).c_str());
            cout << "====== SUBSUITE: " << *sub_suite << " ====== " << endl << endl;
            test_cases = list_directories();
            inner_start_steady = chrono::steady_clock::now();
            double offset_cpu_time = 0.0;
            double offset_wall_time = 0.0;
            for (auto test_case = test_cases.begin(); test_case != test_cases.end(); ++test_case)
            {
                chdir((*test_case).c_str());
                timespec start_offset_cpu_time;
                clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start_offset_cpu_time);
                auto start_offset_wall_time = chrono::steady_clock::now();
                auto pre_compiler = reason_about_test_case();
                timespec end_offset_cpu_time;
                clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end_offset_cpu_time);
                auto end_offset_wall_time = chrono::steady_clock::now();
                offset_cpu_time += (end_offset_cpu_time.tv_sec - start_offset_cpu_time.tv_sec) + (end_offset_cpu_time.tv_nsec - start_offset_cpu_time.tv_nsec) / BILLION;
                offset_wall_time += chrono::duration_cast<chrono::nanoseconds>(inner_end_steady - inner_start_steady).count() / BILLION;
                run_gklee(correct, false_correct, false_incorrect, incorrect, not_supported, pre_compiler);
                cout << " ---- " << *test_case << endl;
                chdir("../.");
            }
            offsets_cpu_time += offset_cpu_time;
            offsets_wall_time += offset_wall_time;
            double inner_wall_time_elapsed =
                chrono::duration_cast<chrono::nanoseconds>(inner_end_steady - inner_start_steady).count() / BILLION - offset_wall_time;
            double inner_cpu_time_elapsed =
                (inner_end.tv_sec - inner_start.tv_sec) + (inner_end.tv_nsec - inner_start.tv_nsec) / BILLION - offset_cpu_time;
            cout << "TIME: [Wall time = " << inner_wall_time_elapsed << "s; CPU time = " << inner_cpu_time_elapsed << "s" << endl;
            cout << "=====================================" << endl;
            test_cases.clear();
            cout << endl;
            chdir("../.");
        }
        sub_suites.clear();
        cout << endl << endl;
        chdir("../.");
    }
    auto end_steady = chrono::steady_clock::now();
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
    double wall_time_elapsed = chrono::duration_cast<chrono::nanoseconds>(end_steady - start_steady).count() / BILLION - offsets_wall_time;
    double cpu_time_elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / BILLION - offsets_cpu_time;
    suites.clear();
    cout << "==========================================" << endl;
    cout << "CORRECT: " << correct << endl;
    cout << "INCORRECT: " << incorrect << endl;
    cout << "FALSE CORRECT: " << false_correct << endl;
    cout << "FALSE INCORRECT: " << false_incorrect << endl;
    cout << "NOT SUPPORTED: " << not_supported << endl;
    cout << "TOTAL TIME: [Wall time = " << wall_time_elapsed << "s; CPU time = " << cpu_time_elapsed << "s" << endl;
    cout << "==========================================" << endl;
}

int main()
{
    calculate_verification_rate();
    return 0;
}

