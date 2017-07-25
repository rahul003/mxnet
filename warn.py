import re
import sys
import operator
from subprocess import check_output, STDOUT, CalledProcessError

def process_output(command_output):
    warnings = {}
    regex = r"(.*):\swarning:\s(.*)"
    lines = command_output.split("\n")
    for line in lines[:-3]:
        matches = re.finditer(regex, line)
        for matchNum, match in enumerate(matches):
            try:
                warnings[match.group()] +=1
            except KeyError:
                warnings[match.group()] =1
    time = lines[-3].split('\t')[1]
    return time, warnings

def generate_stats(warnings):
    total_count = sum(warnings.values())
    sorted_warnings = sorted(warnings.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_warnings, total_count

def print_summary(time, warnings):
    sorted_warnings, total_count = generate_stats(warnings)
    print "START - Compilation warnings count"
    print total_count
    print "END - Compilation warnings count"
    print 'START - Compilation warnings summary'
    print 'Time taken to compile:', time
    print 'Total number of warnings:', total_count, '\n'
    print 'Given below is the list of unique warnings and the number of occurences of that warning'
    for warning, count in sorted_warnings:
        print count, ': ', warning
    print 'END - Compilation warnings summary'

try:
    check_output(['make','clean'], stderr=STDOUT, shell=True)
    command_output = check_output(['time','make','-j8'], stderr=STDOUT, shell=True)
    time, warnings = process_output(command_output)
    print_summary(time, warnings)
except CalledProcessError as ex:
    if ex.returncode > 1:
        print 'Compilation failed'
        print ex.output
        sys.exit(ex.returncode)
