

class Employee:

    raise_amount = 1.04

    def __init__(self, first, last, pay):
        self.first = first
        self.last = last
        self.email = first + '.' + last + '@email.com'
        self.pay = pay


    def fullname(self):
        return '{0} {1}'.format(self.first, self.last)


    def apply_raise(self):
        self.pay = self.pay * self.raise_amount


class Developer(Employee):
    raise_amount = 1.1

    def __init__(self, first, last, pay, prog_lang):
        Employee.__init__(self, first, last, pay)
        self.prog_lang = prog_lang


dev_1 = Employee('Corey', 'Schafer', 50000)
dev_2 = Developer('Test', 'Employee', 60000, 'java')

print(dev_1.email)
print(dev_1.fullname())
print(dev_1.pay)

dev_1.apply_raise()

print(dev_1.pay)
print(Employee.__dict__)
print(dev_2.prog_lang)

#to see the method resolution order
print(help(Developer))