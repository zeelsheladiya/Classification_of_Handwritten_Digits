from hstest.stage_test import StageTest
from hstest.test_case import TestCase
from hstest.check_result import CheckResult
import re


class CCATest(StageTest):

    def generate(self):
        return [TestCase(time_limit=1800000)]

    def check(self, reply, attach):
        lines = reply.split('\n')
        if "" in lines:
            lines = list(filter(lambda a: a != "", lines))

        # general
        lines2check = []

        for item in lines:
            if any(key_word in item.replace(" ", "").lower() for key_word in ['k-nearest', 'forestalgorithm', 'accuracy']):
                lines2check.append(item)

        if len(lines2check) != 4:
            return CheckResult.wrong(
                feedback='Something is wrong with the output format, check the example output at the stage 5')

        # k-nearest neighbours classifier
        algorithm_name_reply = lines2check[0]
        accuracy_reply = re.findall(r'\d*\.\d+|\d+', lines2check[1])
        if len(accuracy_reply) != 1:
            return CheckResult.wrong(feedback='It should be one number in the "accuracy:" section')

        if not 0.957 <= float(accuracy_reply[0]) < 1:
            return CheckResult.wrong(
                feedback=f"The accuracy for {algorithm_name_reply} is wrong")

        # random forest classifier
        algorithm_name_reply = lines2check[2]
        accuracy_reply = re.findall(r'\d*\.\d+|\d+', lines2check[3])
        if len(accuracy_reply) != 1:
            return CheckResult.wrong(feedback='It should be one number in the "accuracy:" section')

        if not 0.945 <= float(accuracy_reply[0]) < 1:
            return CheckResult.wrong(
                feedback=f"The accuracy for {algorithm_name_reply} is wrong")
        return CheckResult.correct()


if __name__ == '__main__':
    CCATest().run_tests()