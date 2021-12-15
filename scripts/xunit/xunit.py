"""Simple xunit results file creator utility."""
# Copyright (c) 2020-2022, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
import xml.etree.ElementTree as ET
from xml.dom import minidom


class xunit_results:
    """Xunit results writer."""

    def __init__(self):
        """Initialize results."""
        self.name = "testsuites"
        self.suites = []

    def create_suite(self, name):
        """Create xunit suite for results."""
        s = xunit_suite(name)
        self.suites.append(s)
        return s

    def write_results(self, filename):
        """Write the results to the appropriate suites."""
        suites = ET.Element(self.name)
        tree = ET.ElementTree(suites)
        for s in self.suites:
            testsuite = ET.SubElement(
                suites, "testsuite", {"name": s.name, "errors": "0"}
            )
            tests = 0
            failures = 0
            skip = 0
            for t in s.tests:
                test = ET.SubElement(
                    testsuite,
                    "testcase",
                    {"name": t.name, "classname": t.classname, "time": t.time},
                )
                tests += 1
                if t.skip:
                    skip += 1
                    ET.SubElement(test, "skipped", {"type": "Skipped test"})
                if t.fail:
                    failures += 1
                    fail = ET.SubElement(test, "failure", {"type": "Test failed"})
                    fail.text = t.fail
                if t.sysout:
                    sysout = ET.SubElement(test, "system-out")
                    sysout.text = t.sysout
                if t.syserr:
                    syserr = ET.SubElement(test, "system-err")
                    syserr.text = t.syserr
            testsuite.attrib["tests"] = str(tests)
            testsuite.attrib["failures"] = str(failures)
            testsuite.attrib["skip"] = str(skip)
        xmlstr = minidom.parseString(ET.tostring(tree.getroot())).toprettyxml(
            indent="  "
        )
        with open(filename, "w") as f:
            f.write(xmlstr)


class xunit_suite:
    """Xunit suite for test results."""

    def __init__(self, name):
        """Initialize suite."""
        self.name = name
        self.tests = []


# classname should be of the form suite.class/subclass/subclass2/...
# You can have an unlimited number of subclasses in this manner


class xunit_test:
    """Xunit test result."""

    def __init__(self, name, classname=None):
        """Initialize test."""
        self.name = name
        if classname:
            self.classname = classname
        else:
            self.classname = name
        self.time = "0.000"
        self.fail = None
        self.skip = False
        self.sysout = None
        self.syserr = None

    def failed(self, text):
        """Set test failed information."""
        self.fail = text

    def skipped(self):
        """Set test as skipped."""
        self.skip = True


if __name__ == "__main__":
    # Simple test
    r = xunit_results()
    s = r.create_suite("selftest")
    for i in range(0, 10):
        t = xunit_test("atest" + str(i), "selftest")
        if i == 3:
            t.failed("Unknown failure foo")
        if i == 7:
            t.skipped()
        s.tests.append(t)
    r.write_results("foo.xml")
