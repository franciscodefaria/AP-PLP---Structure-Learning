#import phcpy
# import coopr.pyomo
import codecs
import sys
import shutil
import os
import fileinput
import re
import copy
import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution, basinhopping
import itertools

class example:
    def __init__(self):
        self.observations=[]

class observation:
    def __init__(self):
        self.variable=None
        self.value=None

class rule:
    def __init__(self, head, body):
        self.head=head
        self.body=body
        self.representation=[]
        self.probability=0.0

class parents_set:
    def __init__(self,variable,variables):
        self.variable=variable
        self.variables=variables
        self.rules=[]

        self.combinations=[]
        self.pruned_combinations=[]

        self.combinations_groups=[]
        self.pruned_combinations_groups=[]

        self.best_combination=rules_combination(variable,[])
        self.score=0.0
        self.accuracy=0.0

    def generate_rules(self,mode):

        if mode!="Naive":

            aux_predicates=[]
            for k_variable in self.variables:
                aux_predicates.append(k_variable.positive_literal)
                aux_predicates.append(k_variable.negative_literal)

            self.rules.append(rule(i_variable.name,tuple(["Empty"])))

            candidate_bodies=[]
            for k in xrange(1,max_body_len+1):
                candidate_bodies.extend(list(itertools.combinations(aux_predicates,k)))

            for k_body in candidate_bodies:
                valid_rule=True
                for l,l_literal in enumerate(k_body):
                    for m,m_literal in enumerate(k_body):
                        if l_literal=="/+"+m_literal or m_literal=="/+"+l_literal:
                            if l!=m:
                                valid_rule=False
                                break
                    if valid_rule==False:
                        break
                if valid_rule:
                    self.rules.append(rule(i_variable.name,k_body))

        else:

            if len(self.variables)==0:
                self.rules.append(rule(i_variable.name,tuple(["Empty"])))

            else:
                candidate_bodies=[]
                for k_variable in self.variables:
                    if len(candidate_bodies)==0:
                        candidate_bodies.append([k_variable.positive_literal])
                        candidate_bodies.append([k_variable.negative_literal])
                    else:
                        aux_bodies=[]
                        for candidate_body in candidate_bodies:
                            aux_body=copy.deepcopy(candidate_body)
                            aux_body.append(k_variable.positive_literal)
                            aux_bodies.append(aux_body)

                            aux_body=copy.deepcopy(candidate_body)
                            aux_body.append(k_variable.negative_literal)
                            aux_bodies.append(aux_body)

                        candidate_bodies=aux_bodies

                for k_body in candidate_bodies:
                    self.rules.append(rule(i_variable.name,k_body))

    def generate_combinations(self,mode):

        if mode!="Naive":
            for j in range(max_rules_combination_size):
                combinations=[]
                combinations.extend(list(itertools.combinations(self.rules,j+1)))

                indexes_list=[]
                for k,k_combination in enumerate(combinations):
                    for l,l_variable in enumerate(self.variables):
                        for m,m_rule in enumerate(k_combination):
                            found_variable=False
                            for n, n_literal in enumerate(m_rule.body):
                                if n_literal==l_variable.name or n_literal=="/+"+l_variable.name:
                                    found_variable=True
                                    break
                            if found_variable:
                                break
                        if not found_variable:
                            indexes_list.append(k)
                            break

                for k in sorted(indexes_list, reverse=True):
                    del combinations[k]

                combinations_group=[]
                for k_combination in combinations:
                    combinations_group.append(rules_combination(i_variable,k_combination))

                self.combinations_groups.append(combinations_group)
                self.combinations.extend(combinations_group)

            #Add combination of 4 mutually disjunctive rules
            if len(j_parents_set.variables)==2:

                aux_rules=[]
                for k,k_rule in enumerate(self.rules):
                    if len(k_rule.body)==len(self.variables):
                        aux_rules.append(k_rule)

                combinations_group=[]
                combinations_group.append(rules_combination(i_variable,aux_rules))

                self.combinations_groups.append(combinations_group)
                self.combinations.extend(combinations_group)

        else:

            j=2**len(self.variables)-1
            combinations=[]
            combinations.extend(list(itertools.combinations(self.rules,j+1)))

            indexes_list=[]
            for k,k_combination in enumerate(combinations):
                for l,l_variable in enumerate(self.variables):
                    if l_variable.name!="Empty":
                        for m,m_rule in enumerate(k_combination):
                            found_variable=False
                            for n, n_literal in enumerate(m_rule.body):
                                if n_literal==l_variable.name or n_literal=="/+"+l_variable.name:
                                    found_variable=True
                                    break
                            if found_variable:
                                break
                        if not found_variable:
                            indexes_list.append(k)
                            break

            for k in sorted(indexes_list, reverse=True):
                del combinations[k]

            combinations_group=[]
            for k_combination in combinations:
                combinations_group.append(rules_combination(i_variable,k_combination))

            self.combinations_groups.append(combinations_group)
            self.combinations.extend(combinations_group)

class variable:
    def __init__(self, name, nickname):
        self.name=name #ok
        self.nickname=nickname
        self.positive_literal=name
        self.negative_literal="/+"+name
        self.parents_set_candidates=[] #ok
        self.parents_set=parents_set(self,[])
        self.priori=0.0
        self.other_variables=[]

    def calculate_priori(self,examples):

        #Estimate a priori probability
        self.priori=0.0
        for i, i_example in enumerate(examples):
            for l, l_observation in enumerate(i_example.observations):
                if l_observation.variable.name==self.name:
                    if l_observation.value==True:
                        self.priori+=1.0
        self.priori=self.priori/len(examples)

class rules_combination:
    def __init__(self,variable,combination):
        self.variable=variable
        self.rules=combination

        self.expression=[]
        self.pattern=""

        self.keep_pattern=False
        self.description_length=0.0

        self.occurences=[]
        self.configurations=[]
        self.body_configurations=[]
        self.multipliers=[]

        self.probabilities_list=[0.0]*len(combination)
        self.score=0.0

    def accuracy(self, parameters,cutoff):

        TP_TN=0.0
        FP_FN=0.0
        #print "                Pprobabtilities:", parameters
        for i, i_configuration in enumerate(self.configurations):
            #print "                    self.configurations[i]=", self.configurations[i]
            configuration_likelihood=0.0
            #print "                    self.multipliers[i]=", self.multipliers[i]
            for j,j_rule in enumerate(self.rules):
                #print "                            self.multipliers[i][j])*parameters[j]=", abs(self.multipliers[i][j])*parameters[j]
                #print "                            configuration_likelihood=",configuration_likelihood
                configuration_likelihood=abs(self.multipliers[i][j])*parameters[j]+configuration_likelihood-abs(self.multipliers[i][j])*parameters[j]*configuration_likelihood
                #print "                            -> partial configuration likelihood:", configuration_likelihood
            if i_configuration[len(i_configuration)-1]==-1:
                configuration_likelihood=1.0-configuration_likelihood
            #print "                        self.occurences[i]:",self.occurences[i]
            #print "                        configuration likelihood:", configuration_likelihood
            if configuration_likelihood>=cutoff:
                TP_TN+=1
            else:
                FP_FN+=1

        accuracy=TP_TN/(TP_TN+FP_FN)

        return accuracy

    def minus_log_likelihood(self, parameters):

        log_likelihood=0.0
        #print "                Pprobabtilities:", parameters
        for i, i_configuration in enumerate(self.configurations):
            #print "                    self.configurations[i]=", self.configurations[i]
            configuration_likelihood=0.0
            #print "                    self.multipliers[i]=", self.multipliers[i]
            for j,j_rule in enumerate(self.rules):
                #print "                            self.multipliers[i][j])*parameters[j]=", abs(self.multipliers[i][j])*parameters[j]
                #print "                            configuration_likelihood=",configuration_likelihood
                configuration_likelihood=abs(self.multipliers[i][j])*parameters[j]+configuration_likelihood-abs(self.multipliers[i][j])*parameters[j]*configuration_likelihood
                #print "                            -> partial configuration likelihood:", configuration_likelihood
            if i_configuration[len(i_configuration)-1]==-1:
                configuration_likelihood=1.0-configuration_likelihood
            #print "                        self.occurences[i]:",self.occurences[i]
            #print "                        configuration likelihood:", configuration_likelihood
            log_likelihood=log_likelihood+self.occurences[i]*math.log10(configuration_likelihood)
            #print "                        log-likelihood:", log_likelihood

        return -1*log_likelihood

    def calculate_multipliers(self):

        if len(self.configurations)==2:
            self.multipliers=self.configurations
        else:

            for i, i_configuration in enumerate(self.configurations):
                new_multipliers=[0]*len(self.rules)
                for j, j_body_configuration in enumerate(self.body_configurations):

                    if i_configuration[:len(i_configuration)-1]==j_body_configuration: #activates rule

                        if i_configuration[len(i_configuration)-1]==True:
                            for k in self.expression[j].split("r"):
                                if k!="":
                                    new_multipliers[int(k)-1]=1 #rule is true
                        else:
                            for k in self.expression[j].split("r"):
                                if k!="":
                                    new_multipliers[int(k)-1]=-1 #rule is false

                self.multipliers.append(new_multipliers)

        print "                multipliers=", self.multipliers


    def sollution(self):

        #Combinations of 1 rule
        if len(self.expression)==0:
            A1=self.occurences[0]
            A0=self.occurences[1]
            print "            A0=",A0,", A1=",A1

            aux=np.float64(A0)/(A0+A1)
            if A0+A1==0:
                aux=0.5
            self.probabilities_list[0]=max(min(aux,0.999),0.001)#r1

            return True

        if set(["r1"])==set(self.expression)-set(["0"])-set([""]):
            coefficients=self.calculate_coefficients(["r1"])
            A1=coefficients[0]
            A0=coefficients[1]
            print "            A0=",A0,",A1=",A1

            aux=np.float64(A0)/(A0+A1)
            if A0+A1==0:
                aux=0.5
            self.probabilities_list[0]=max(min(aux,0.999),0.001)#r1

            return True

        #Combinations of 2 rules
        if set(["r1","r2"])==set(self.expression)-set(["0"]):
            coefficients=self.calculate_coefficients(["r1","r2"])
            A1=coefficients[0]
            A0=coefficients[1]
            B1=coefficients[2]
            B0=coefficients[3]
            print "            A0=",A0,",A1=",A1
            print "            B0=",B0,",B1=",B1

            aux=np.float64(A0)/(A0+A1)
            if A0+A1==0:
                aux=0.5
            aux_a=max(min(aux,0.999),0.001) #r1

            aux=np.float64(B0)/(B0+B1)
            if B0+B1==0:
                aux=0.5
            aux_b=max(min(aux,0.999),0.001) #r1

            self.probabilities_list[0]=aux_a #r1
            self.probabilities_list[1]=aux_b #r2

            return True

        if set(["r1","r1r2"])==set(self.expression)-set(["0"]):
            coefficients=self.calculate_coefficients(["r1","r1r2"])
            A1=coefficients[0]
            A0=coefficients[1]
            B1=coefficients[2]
            B0=coefficients[3]
            print "            A0=",A0,",A1=",A1
            print "            B0=",B0,",B1=",B1

            exact_sollution=True

            aux=np.float64(A0)/(A0+A1)
            aux_a=max(min(aux,0.999),0.001) #r1

            aux=np.float64(A1*B0-A0*B1)/(A1*B0+A1*B1)
            aux_b=max(min(aux,0.999),0.001) #r2

            if A0+A1==0:
                aux_a=0.0
                aux_b=np.float64(B0)/(B0+B1)
                if B0+B1==0:
                    aux_a=0.5
                    aux_b=0.5

            if A1*B0+A1*B1==0:
                if A1==0:
                    if B0==0 and B1==0:
                        aux_b=0.5
                    if B0==0 and B1!=0:
                        aux_b=0.0
                    if B0!=0 and B1==0:
                        aux_b=1.0
                    if B0!=0 and B1!=0:
                        aux_b=B0/(B0+B1)
                        if exact_sollution==True:
                            exact_sollution=False
                else: #A1!=0
                    if B0==0 and B1==0:
                        aux_b=0.5

            self.probabilities_list[0]=aux_a #r1
            self.probabilities_list[1]=aux_b #r2

            if not exact_sollution:
                self.probabilities_list=self.numerical_sollution()

            return True

        if set(["r1","r2","r1r2"])==set(self.expression)-set(["0"]): #No exact sollution
            coefficients=self.calculate_coefficients(["r1","r2","r1r2"])
            A1=coefficients[0]
            A0=coefficients[1]
            B1=coefficients[2]
            B0=coefficients[3]
            C1=coefficients[4]
            C0=coefficients[5]
            print "            A0=",A0,",A1=",A1
            print "            B0=",B0,",B1=",B1
            print "            C0=",C0,",C1=",C1

            aux_a=np.float64(A0+C0)/(A0+A1+C0+C1)
            if A0+A1+C0+C1==0:
                aux_a=0.5

            aux_b=np.float64(B0+C0)/(B0+B1+C0+C1)
            if B0+B1+C0+C1==0:
                aux_b=0.5

            self.probabilities_list[0]=aux_a
            self.probabilities_list[1]=aux_b

            self.probabilities_list=self.numerical_sollution()

            return True

        #Combinations of 3 rules
        if set(["r1","r2","r3"])==set(self.expression)-set(["0"]):
            coefficients=self.calculate_coefficients(["r1","r2","r3"])
            A1=coefficients[0]
            A0=coefficients[1]
            B1=coefficients[2]
            B0=coefficients[3]
            C1=coefficients[4]
            C0=coefficients[5]
            print "            A0=",A0,",A1=",A1
            print "            B0=",B0,",B1=",B1
            print "            C0=",C0,",C1=",C1

            aux=np.float64(A0)/(A0+A1)
            if A0+A1==0:
                aux=0.5
            aux_a=max(min(aux,0.999),0.001)#r1

            aux=np.float64(B0)/(B0+B1)
            if B0+B1==0:
                aux=0.5
            aux_b=max(min(aux,0.999),0.001)#r2

            aux=np.float64(C0)/(C0+C1)
            if C0+C1==0:
                aux=0.5
            aux_c=max(min(aux,0.999),0.001) #r3

            self.probabilities_list[0]=aux_a
            self.probabilities_list[1]=aux_b
            self.probabilities_list[2]=aux_c

            return True

        if set(["r1","r2","r1r2","r3"])==set(self.expression)-set(["0"]): #No exact sollution
            coefficients=self.calculate_coefficients(["r1","r2","r1r2","r3"])
            A1=coefficients[0]
            A0=coefficients[1]
            B1=coefficients[2]
            B0=coefficients[3]
            C1=coefficients[4]
            C0=coefficients[5]
            D1=coefficients[6]
            D0=coefficients[7]
            print "            A0=",A0,",A1=",A1
            print "            B0=",B0,",B1=",B1
            print "            C0=",C0,",C1=",C1
            print "            D0=",D0,",D1=",D1

            aux_a=np.float64(A0+C0)/(A0+A1+C0+C1)
            if A0+A1+C0+C1==0:
                aux_a=0.5

            aux_b=np.float64(B0+C0)/(B0+B1+C0+C1)
            if B0+B1+C0+C1==0:
                aux_b=0.5

            aux_c=np.float64(D0)/(D0+D1)
            if D0+D1==0:
                aux_c=0.5

            self.probabilities_list[0]=aux_a
            self.probabilities_list[1]=aux_b
            self.probabilities_list[2]=aux_c

            self.probabilities_list=self.numerical_sollution()

        #     return True

        if set(["r1","r1r2","r1r3","r1r2r3"])==set(self.expression)-set(["0"]): #No exact sollution
            coefficients=self.calculate_coefficients(["r1","r1r2","r1r3","r1r2r3"])
            A1=coefficients[0]
            A0=coefficients[1]
            B1=coefficients[2]
            B0=coefficients[3]
            C1=coefficients[4]
            C0=coefficients[5]
            D1=coefficients[6]
            D0=coefficients[7]
            print "            A0=",A0,",A1=",A1
            print "            B0=",B0,",B1=",B1
            print "            C0=",C0,",C1=",C1
            print "            D0=",D0,",D1=",D1

            aux_a=np.float64(A0+B0+C0+D0)/(A0+A1+B0+B1+C0+C1+D0+D1)
            if A0+A1+B0+B1+C0+C1+D0+D1==0:
                aux_a=0.5

            aux_b=np.float64(B0+D0)/(B0+B1+D0+D1)
            if C0+C1+D0+D1==0:
                aux_b=0.5

            aux_c=np.float64(C0+D0)/(C0+C1+D0+D1)
            if C0+C1+D0+D1==0:
                aux_c=0.5

            self.probabilities_list[0]=aux_a
            self.probabilities_list[1]=aux_b
            self.probabilities_list[2]=aux_c

            self.probabilities_list=self.numerical_sollution()

        if set(["r1","r2","r1r3","r2r3"])==set(self.expression)-set(["0"]): #No exact sollution
            coefficients=self.calculate_coefficients(["r1","r2","r1r3","r2r3"])
            A1=coefficients[0]
            A0=coefficients[1]
            B1=coefficients[2]
            B0=coefficients[3]
            C1=coefficients[4]
            C0=coefficients[5]
            D1=coefficients[6]
            D0=coefficients[7]
            print "            A0=",A0,",A1=",A1
            print "            B0=",B0,",B1=",B1
            print "            C0=",C0,",C1=",C1
            print "            D0=",D0,",D1=",D1

            aux_a=np.float64(A0+C0)/(A0+A1+C0+C1)
            if A0+A1+C0+C1==0:
                aux_a=0.5

            aux_b=np.float64(B0+D0)/(B0+B1+D0+D1)
            if B0+B1+D0+D1==0:
                aux_b=0.5

            aux_c=np.float64(C0+D0)/(C0+C1+D0+D1)
            if C0+C1+D0+D1==0:
                aux_c=0.5

            self.probabilities_list[0]=aux_a
            self.probabilities_list[1]=aux_b
            self.probabilities_list[2]=aux_c

            self.probabilities_list=self.numerical_sollution()

        if set(["r2","r3","r1r2","r1r3"])==set(self.expression)-set(["0"]): #No exact sollution
            coefficients=self.calculate_coefficients(["r2","r3","r1r2","r1r3"])
            A1=coefficients[0]
            A0=coefficients[1]
            B1=coefficients[2]
            B0=coefficients[3]
            C1=coefficients[4]
            C0=coefficients[5]
            D1=coefficients[6]
            D0=coefficients[7]
            print "            A0=",A0,",A1=",A1
            print "            B0=",B0,",B1=",B1
            print "            C0=",C0,",C1=",C1
            print "            D0=",D0,",D1=",D1

            aux_a=np.float64(C0+D0)/(C0+C1+D0+D1)
            if C0+C1+D0+D1==0:
                aux_a=0.5

            aux_b=np.float64(A0+C0)/(A0+A1+C0+C1)
            if A0+A1+C0+C1==0:
                aux_b=0.5

            aux_c=np.float64(B0+D0)/(B0+B1+D0+D1)
            if B0+B1+D0+D1==0:
                aux_c=0.5

            self.probabilities_list[0]=aux_a
            self.probabilities_list[1]=aux_b
            self.probabilities_list[2]=aux_c

            self.probabilities_list=self.numerical_sollution()

        if set(["r1","r1r2","r1r3"])==set(self.expression)-set(["0"]): #Ok
            coefficients=self.calculate_coefficients(["r1","r1r2","r1r3"])
            A1=coefficients[0]
            A0=coefficients[1]
            B1=coefficients[2]
            B0=coefficients[3]
            C1=coefficients[4]
            C0=coefficients[5]
            print "            A0=",A0,",A1=",A1
            print "            B0=",B0,",B1=",B1
            print "            C0=",C0,",C1=",C1

            exact_sollution=True

            aux=np.float64(A0)/(A0+A1)
            if A0+A1==0:
                aux=0.5
            aux_a=max(min(aux,0.99),0.001)

            aux=np.float64(A1*B0-A0*B1)/(A1*B0+A1*B1)
            aux_b=max(min(aux,0.99),0.001)

            aux=np.float64(A1*C0-A0*C1)/(A1*C0+A1*C1)
            aux_c=max(min(aux,0.99),0.001)

            if A1*B0+A1*B1==0 and A1*C0+A1*C1==0:
                if A1==0:
                    if B0==0 and B1==0:
                        aux_b=0.5
                    if B0==0 and B1!=0:
                        aux_b=0.0
                    if B0!=0 and B1==0:
                        aux_b=1.0
                    if B0!=0 and B1!=0:
                        aux_b=B0/(B0+B1)
                        if exact_sollution==True:
                            exact_sollution=False

                    if C0==0 and C1==0:
                        aux_c=0.5
                    if C0==0 and C1!=0:
                        aux_c=0.0
                    if C0!=0 and C1==0:
                        aux_c=1.0
                    if C0!=0 and C1!=0:
                        aux_c=B0/(C0+C1)
                        if exact_sollution==True:
                            exact_sollution=False

                else:# -> B0=B1=C0=C1=0
                    aux_b=0.5
                    aux_c=0.5

            if A1*B0+A1*B1==0 and A1*C0+A1*C1!=0:# -> B0=B1=0
                aux_b=0.5

            if A1*B0+A1*B1!=0 and A1*C0+A1*C1==0:# -> C0=C1=0
                aux_c=0.5

            self.probabilities_list[0]=aux_a #r1
            self.probabilities_list[1]=aux_b #r2
            self.probabilities_list[2]=aux_c #r3

            if not exact_sollution:
                self.probabilities_list=self.numerical_sollution()

        #Combinations of 4 rules
        if set(["r1","r2","r3","r4"])==set(self.expression)-set(["0"]): #Ok
            coefficients=self.calculate_coefficients(["r1","r2","r3","r4"])
            A1=coefficients[0]
            A0=coefficients[1]
            B1=coefficients[2]
            B0=coefficients[3]
            C1=coefficients[4]
            C0=coefficients[5]
            D1=coefficients[6]
            D0=coefficients[7]
            print "            A0=",A0,",A1=",A1
            print "            B0=",B0,",B1=",B1
            print "            C0=",C0,",C1=",C1
            print "            D0=",D0,",D1=",D1

            aux_a=np.float64(A0)/(A0+A1)
            if A0+A1==0:
                aux_a=0.5

            aux_b=np.float64(B0)/(B0+B1)
            if B0+B1==0:
                aux_b=0.5

            aux_c=np.float64(C0)/(C0+C1)
            if C0+C1==0:
                aux_c=0.5

            aux_d=np.float64(D0)/(D0+D1)
            if D0+D1==0:
                aux_d=0.5

            self.probabilities_list[0]=max(min(aux_a,0.999),0.001) #r1
            self.probabilities_list[1]=max(min(aux_b,0.999),0.001) #r2
            self.probabilities_list[2]=max(min(aux_c,0.999),0.001) #r3
            self.probabilities_list[3]=max(min(aux_d,0.999),0.001) #r4

            print "            Parameters=", self.probabilities_list

            return True


        return False

    def calculate_coefficients(self,pattern_list):
        coefficients=[]
        for i, i_element in enumerate(pattern_list):
            coefficients.extend([0.0,0.0])
            for j, j_element in enumerate(self.expression):
                if j_element==i_element:
                    coefficients[2*i]+=self.occurences[2*j]
                    coefficients[2*i+1]+=self.occurences[2*j+1]
        #print "                coefficents=",coefficients
        return coefficients

    def initialize_parameters(self):

        found_nan=False
        for p,probability_value in enumerate(self.probabilities_list):
            if math.isnan(probability_value):
                found_nan=True
                print "\a"
            aux=[0.0,0.0]
            for q,q_item in enumerate(self.expression):
                if str(p+1)in q_item:
                    aux[0]+=self.occurences[2*q+1]
                    aux[1]+=(self.occurences[2*q]+self.occurences[2*q+1])
            self.probabilities_list[p]=max(min(np.float64(aux[0])/aux[1],0.999),0.001)
            # if math.isnan(probability_value):
            #     aux=[0.0,0.0]
            #     for q,q_item in enumerate(self.expression):
            #         if str(p+1)in q_item:
            #             aux[0]+=self.occurences[2*q+1]
            #             aux[1]+=(self.occurences[2*q]+self.occurences[2*q+1])
            #     self.probabilities_list[p]=np.float64(aux[0])/aux[1]
            #     found_nan=True

        if not found_nan:
            for p,probability_value in enumerate(self.probabilities_list):
                aux=[0.0,0.0]
                for q,q_item in enumerate(self.expression):
                    if str(p+1)in q_item:
                        aux[0]+=self.occurences[2*q]
                        aux[1]+=(self.occurences[2*q]+self.occurences[2*q+1])
                self.probabilities_list[p]=max(min(np.float64(aux[0])/aux[1],0.999),0.001)
                if math.isnan(self.probabilities_list[p]):
                    self.probabilities_list[p]=0.5

    def numerical_sollution(self):
        print "          * Previous values of parameters =", self.probabilities_list
        self.initialize_parameters()
        print "          * Numerical sollution with initial parameters =", self.probabilities_list
        print "          * Log-likelihood before solving: ",-1*self.minus_log_likelihood(self.probabilities_list)
        #self.probabilities_list=objective(rule=self.minus_likelihood)

        #Numerical approximation for combinations without exact sollution
        bnds=[(0.001,0.999)]*len(self.probabilities_list)
        #minimize_return=minimize(self.minus_log_likelihood, self.probabilities_list, method='L-BFGS-B', bounds=bnds, options={'disp': True ,'eps' : 1e-7})
        minimizer_kwargs = dict(method="L-BFGS-B", bounds=bnds)
        minimize_return = basinhopping(self.minus_log_likelihood, self.probabilities_list, minimizer_kwargs=minimizer_kwargs)

        self.probabilities_list=minimize_return.x

        # #Using PHC
        # input_file=open("phc_input.txt", 'w')
        # input_file.write(str(len(equations))+ '\n')
        # for equation in equations:
        #     input_file.write("    "+equation+ ";\n")
        #     #print "    "+equation
        # input_file.close
        # #os.system("./phc -b phc_input.txt phc_output.txt")

        return self.probabilities_list




print "\n##############"
print "#Pre-Settings#"
print "##############"

max_rules_combination_size=3
max_parents_set_size=2
max_body_len=2#4

print "\n###############################"
print "#Identify variables in dataset#"
print "###############################"

file_name=str(sys.argv[1])
with codecs.open(file_name, "r", encoding="utf-8-sig") as f:
    i=0
    observed_variables=[]
    for i_line in f:

        i_example=i_line.strip().split(",")

        for j_observation in i_example:

            if "/+" in j_observation:
                aux_observation=str(j_observation[2:])
            else:
                aux_observation=str(j_observation)

            if aux_observation not in [o.name for o in observed_variables]:
                observed_variables.append(variable(aux_observation,len(observed_variables)))

for i_variable in observed_variables:
	print "\nVariable:", i_variable.name
	print "    Precicates:", i_variable.positive_literal,",",i_variable.negative_literal

print "\n########################"
print "#Store dataset examples#"
print "########################"

file_name=str(sys.argv[1])
with codecs.open(file_name, "r", encoding="utf-8-sig") as f:

    examples=[]
    for i_line in f:
        i_example=i_line.split(",")
        #print "i_example=", i_example
        new_example=example()
        for j_observation in i_example:
            new_observation=observation()

            if "/+" in j_observation:
                new_observation.value=False
                aux_observation=str(j_observation[2:])
                aux_observation=aux_observation.replace("\r", "")
                aux_observation=aux_observation.replace("\n", "")
            else:
                new_observation.value=True
                aux_observation=str(j_observation)
                aux_observation=aux_observation.replace("\r", "")
                aux_observation=aux_observation.replace("\n", "")

            for k,k_variable in enumerate(observed_variables):
                if k_variable.name==aux_observation:
                    new_observation.variable=k_variable
                    break

            new_example.observations.append(new_observation)

        examples.append(new_example)

print "\nExamples:"
for i,i_example in enumerate(examples):
    print i+1,":",[[o.value, o.variable.name] for o in i_example.observations]

print "\n###############################"
print "#Build examples configurations#"
print "###############################"

examples_configurations=[]
examples_occurence=[]

for i,i_example in enumerate(examples):
    new_configuration=[]
    for j,j_observation in enumerate(i_example.observations):
        if j_observation.value==True:
            new_configuration.append(1)
        if j_observation.value==False:
            new_configuration.append(0)

    found_configuration=False
    for j, j_configuration in enumerate(examples_configurations):
        if j_configuration==new_configuration:
            found_configuration=True
            break
    if not found_configuration:
        examples_configurations.append(new_configuration)
        examples_occurence.append(1)
    else:
        examples_occurence[j]+=1

for i,i_example_configuration in enumerate(examples_configurations):
    print i+1,":",i_example_configuration, " -> N=",examples_occurence[i]

print "\n############################"
print "#Calculate variables priori#"
print "############################"

for i_variable in observed_variables:
    i_variable.calculate_priori(examples)

print "\n############################"
print "#Create auxiliary variables#"
print "############################"
variables=[]
for i in range(max_parents_set_size+1):
    variables.append(variable("aux"+str(i),"aux"+str(i)))
for i_variable in variables:
    print i_variable.name

print "\n###################"
print "#Build parents_sets#"
print "###################"
i=0
i_variable=variables[0]
i_variable.other_variables=variables[1:]
#print "Obther variables: ",[o.name for o in i_variable.other_variables]

#Create parents set candidates for variable
for j in range(max_parents_set_size+1):
    other_variables_combinations=list(itertools.combinations(i_variable.other_variables,j))
    for other_variables_combination in other_variables_combinations:
        i_variable.parents_set_candidates.append(parents_set(i_variable,other_variables_combination))

#print "Variable: ",i_variable.name
for j, j_parents_set in enumerate(i_variable.parents_set_candidates):
    print "    \n\nParents:", [o.name for o in j_parents_set.variables]

    print "\n        ##############"
    print "        #Create rules#"
    print "        ##############"

    mode=str(sys.argv[4])
    j_parents_set.generate_rules(mode)

    for k, k_rule in enumerate(j_parents_set.rules):
        print "        ",k_rule.head, ":-",k_rule.body," -> Representation=",k_rule.representation

    print "\n        ###############################################################"
    print "        #Build all possible combinations of rules for each parents set#"
    print "        ###############################################################"

    j_parents_set.generate_combinations(mode)

    for k,k_combinations_group in enumerate(j_parents_set.combinations_groups):
        print "        Group", k+1
        for l,l_combination in enumerate(k_combinations_group):
            print "            Combination ",l+1, "from Group", k+1
            for m_rule in l_combination.rules: print "                ",m_rule.head, ":-",m_rule.body

    print "\n        ######################################"
    print "        #Find expression for each combination#"
    print "        ######################################"

    body_configurations=[]
    configurations=[[-1],[1]]
    if len(j_parents_set.variables)>0:
        for k in range(len(j_parents_set.variables)):
            if k==len(j_parents_set.variables)-1:
                body_configurations=copy.deepcopy(configurations)

            new_configurations=[]
            for l_configuration in configurations:
                new_configuration=copy.deepcopy(l_configuration)
                new_configuration.append(-1)
                new_configurations.append(new_configuration)

                new_configuration=copy.deepcopy(l_configuration)
                new_configuration.append(1)
                new_configurations.append(new_configuration)
            configurations=new_configurations

    # print "        Configurations:", configurations
    # print "        Body configurations:", body_configurations

    for k,k_combination in enumerate(j_parents_set.combinations):

        #Find this combination rules representation
        for l,l_rule in enumerate(k_combination.rules):
            l_rule.representation=[]
            for m_variable in j_parents_set.variables:
                if m_variable!=i_variable:
                    if m_variable.name in l_rule.body and "/+"+m_variable.name not in l_rule.body:
                        l_rule.representation.append(1)
                    if m_variable.name not in l_rule.body and "/+"+m_variable.name in l_rule.body:
                        l_rule.representation.append(-1)
                    if m_variable.name not in l_rule.body and "/+"+m_variable.name not in l_rule.body:
                        l_rule.representation.append(0)

            if l_rule.body==tuple(["Empty"]):
                l_rule.representation.append(1)
            else:
                l_rule.representation.append(0)

        #Find combination expression
        k_combination.configurations=configurations
        k_combination.body_configurations=body_configurations

        for l_configuration in k_combination.body_configurations:

            configuration_expression=""
            for m,m_rule in enumerate(k_combination.rules):
                aux_configuration=copy.deepcopy(l_configuration)
                aux_product=[a*b for a,b in zip(aux_configuration,m_rule.representation)]

                if all(n != -1 for n in aux_product) and m_rule.representation[len(m_rule.representation)-1]!=1:
                    configuration_expression=configuration_expression+"r"+str(m+1)
                if m_rule.representation[len(m_rule.representation)-1]==1:
                    configuration_expression=configuration_expression+"r"+str(m+1)

            k_combination.expression.append(configuration_expression)

        print "        Combination ",k+1, "from parents set", [o.name for o in j_parents_set.variables] ," -> ", k_combination.expression
        for l_rule in k_combination.rules: print "            ",l_rule.head, ":-",l_rule.body," -> Representation=",l_rule.representation

    print "\n        ###############"
    print "        #Find patterns#"
    print "        ###############"

    letters=["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","x","y","w","z"]

    for k, k_combination in enumerate(j_parents_set.combinations):
        aux_letters=copy.deepcopy(letters)
        for l,l_element in enumerate(k_combination.expression):
            #print "l_element=", l_element
            if l_element=="":
                k_combination.pattern=k_combination.pattern+str(0)
            else:
                if l_element in k_combination.expression[:l]:
                    for m,m_element in enumerate(k_combination.expression[:l]):
                        if l_element==m_element:
                            k_combination.pattern=k_combination.pattern+k_combination.pattern[m]
                            break
                else:
                    k_combination.pattern=k_combination.pattern+aux_letters[0]
                    aux_letters.pop(0)

        print "        Combination ",k+1, "from parents set", [o.name for o in j_parents_set.variables]," -> ", k_combination.expression , " == ", k_combination.pattern
        for l_rule in k_combination.rules: print "           ",l_rule.head, ":-",l_rule.body

ref_variable=copy.deepcopy(i_variable)

print "\n###################"
print "#Build parents_sets#"
print "###################"

for i, i_variable in enumerate(observed_variables):
    #print "\nVariable: ", i_variable.name
    other_variables=copy.deepcopy(observed_variables)
    del other_variables[i]
    #print "Other variables:", [o.name for o in other_variables]

    #Create parents set candidates for variable
    other_variables_combinations=list(itertools.combinations(other_variables,max_parents_set_size))
    #print "other_variables_combinations=",[[o.name for o in m] for m in other_variables_combinations]

    aux_variables=[]
    for k,other_variables_combination in enumerate(other_variables_combinations):

        aux_name=i_variable.name
        aux_nickname=i_variable.nickname
        aux_positive_literal=aux_name
        aux_negative_literal="/+"+aux_name

        aux_variable=copy.deepcopy(ref_variable)

        aux_variable.name=copy.deepcopy(aux_name)
        aux_variable.nickname=copy.deepcopy(aux_nickname)
        aux_variable.positive_literal=copy.deepcopy(aux_positive_literal)
        aux_variable.negative_literal=copy.deepcopy(aux_negative_literal)

        for l,other_variable in enumerate(other_variables_combination):
            aux_variable.other_variables[l].name=other_variable.name
            aux_variable.other_variables[l].nickname=other_variable.nickname
        aux_variables.append(aux_variable)

        #print "other_variables_combination=",[o.name for o in other_variables_combination]
        #print "aux_variable.other_variables=",[o.name for o in aux_variable.other_variables]

    i_variable.parents_set_candidates=[]
    for k,aux_variable in enumerate(aux_variables):
        for parents_set_candidate in aux_variable.parents_set_candidates:
            parents_set_candidate.variable=i_variable

        for l,l_parents_set in enumerate(aux_variable.parents_set_candidates):
            #print "    Parents: ",[o.name for o in j_parents_set.variables]

            found_parents_set=False
            for m,m_parents_set in enumerate(i_variable.parents_set_candidates):
                if set([o.name for o in l_parents_set.variables])==set([o.name for o in m_parents_set.variables]):
                    found_parents_set=True
                    break

            if not found_parents_set:
                for n,n_combination in enumerate(l_parents_set.combinations):
                    #print "        Combination ",n+1, "from parents set", [o.name for o in l_parents_set.variables]," -> ", n_combination.expression , " == ", n_combination.pattern
                    aux_rules=[]
                    for o,o_rule in enumerate(n_combination.rules):
                        #print "            Before: ",o_rule.head, ":-",o_rule.body
                        #print "i_variable.name=", i_variable.name
                        aux_head=i_variable.name
                        aux_body=[]
                        for p,p_literal in enumerate(o_rule.body):
                            if len(l_parents_set.variables)==1:
                                if p_literal=="aux1":
                                    aux_body.append(l_parents_set.variables[0].name)
                                if p_literal=="/+aux1":
                                    aux_body.append("/+"+l_parents_set.variables[0].name)
                                if p_literal=="aux2":
                                    aux_body.append(l_parents_set.variables[0].name)
                                if p_literal=="/+aux2":
                                    aux_body.append("/+"+l_parents_set.variables[0].name)
                            if len(l_parents_set.variables)==2:
                                if p_literal=="aux1":
                                    aux_body.append(l_parents_set.variables[0].name)
                                if p_literal=="/+aux1":
                                    aux_body.append("/+"+l_parents_set.variables[0].name)
                                if p_literal=="aux2":
                                    aux_body.append(l_parents_set.variables[1].name)
                                if p_literal=="/+aux2":
                                    aux_body.append("/+"+l_parents_set.variables[1].name)
                        aux_rule=rule(aux_head,aux_body)
                        aux_rules.append(aux_rule)
                        #print "            After: ",aux_rule.head, ":-",aux_rule.body
                    n_combination.rules=aux_rules

                i_variable.parents_set_candidates.append(l_parents_set)

for i, i_variable in enumerate(observed_variables):
    indexes_list=[]
    for j, j_parents_set in enumerate(i_variable.parents_set_candidates):
        if len(j_parents_set.combinations)==0:
            indexes_list.append(j)
    for j in sorted(indexes_list, reverse=True):
        del i_variable.parents_set_candidates[j]

for i, i_variable in enumerate(observed_variables):
    print "\nVariable: ", i_variable.name
    for j, j_parents_set in enumerate(i_variable.parents_set_candidates):
        print "    Parents: ",[o.name for o in j_parents_set.variables]
        for k, k_combination in enumerate(j_parents_set.combinations):
            print "        Combination ",k+1, "from parents set", [o.name for o in j_parents_set.variables]," -> ", k_combination.expression , " == ", k_combination.pattern
            for l_rule in k_combination.rules: print "           ",l_rule.head, ":-",l_rule.body


print "\n##############################################################"
print "#Count patterns and prune among combinations of the same size#"
print "##############################################################"

for i, i_variable in enumerate(observed_variables):
    print "\nVariable: ", i_variable.name
    for j, j_parents_set in enumerate(i_variable.parents_set_candidates):

        #Create list of patterns
        patterns=[]
        print "\n    Parents: ",[o.name for o in j_parents_set.variables]
        for k, k_combination in enumerate(j_parents_set.combinations):
            found_pattern=False
            for l,l_pattern in enumerate(patterns):
                if k_combination.pattern == l_pattern[0]:
                    patterns[l][1]+=1
                    found_pattern=True
                    break
            if not found_pattern:
                patterns.append([k_combination.pattern,1])
        #print "    Patterns: ", patterns

#         pruned_pattern_groups=[]
#         pruned_combinations_group=[]

        #Gather combinations of same pattern
        pattern_groups=[]
        for k,k_pattern in enumerate(patterns):
            pattern_group=[]
            for l,l_combination in enumerate(j_parents_set.combinations):
                if l_combination.pattern == k_pattern[0]:
                    pattern_group.append(l_combination)
            pattern_groups.append(pattern_group)

            print "\n        Pattern=",k_pattern[0]
            print "            Before pruning:"
            for l,l_combination in enumerate(pattern_groups[k]):
                print "                Combination ",l+1, "from pattern", k_pattern[0],":", [o.name for o in j_parents_set.variables]," -> ", l_combination.expression , " == ", l_combination.pattern
                for m_rule in l_combination.rules: print "                   ",m_rule.head, ":-",m_rule.body

            #Calculate description length
            for l,l_combination in enumerate(pattern_groups[k]):
                for m,m_rule in enumerate(l_combination.rules):
                    l_combination.description_length+=0.5*math.log10(len(examples))
                    #l_combination.description_length+=math.log10(2*len(variables))
                    #l_combination.description_length+=math.log10(2*len(variables))*len(m_rule.body)

            #Pruning
            if len(pattern_groups[k])==9:
                pattern_groups[k]=[pattern_groups[k][0],pattern_groups[k][8]]
            else:
                if len(pattern_groups[k])==8:
                    pattern_groups[k]=[pattern_groups[k][7]]
                else:
                    if len(pattern_groups[k])==4:
                        pattern_groups[k]=[pattern_groups[k][2]]

            print "            After pruning:"
            for l,l_combination in enumerate(pattern_groups[k]):
                print "                Combination ",l+1, "from pattern", k_pattern[0],":", [o.name for o in j_parents_set.variables]," -> ", l_combination.expression , " == ", l_combination.pattern
                for m_rule in l_combination.rules: print "                   ",m_rule.head, ":-",m_rule.body

            for l, l_combination in enumerate(pattern_groups[k]):
                j_parents_set.pruned_combinations.append(l_combination)

#Print combinations left
for i, i_variable in enumerate(observed_variables):
    print "\nVariable: ", i_variable.name
    for j, j_parents_set in enumerate(i_variable.parents_set_candidates):
        print "    Parents: ",[o.name for o in j_parents_set.variables]
        for k,k_combination in enumerate(j_parents_set.pruned_combinations):
            print "        Combination ",k+1, "from parents set", [o.name for o in j_parents_set.variables]," -> ", k_combination.expression , " == ", k_combination.pattern
            for l_rule in k_combination.rules: print "        ",l_rule.head, ":-",l_rule.body

print "\n############################################################"
print "#Prune combinaitons that have likelihood zero in the dataset#"
print "#############################################################"

for i, i_variable in enumerate(observed_variables):
    #print "\nVariable: ", i_variable.name
    for j, j_parents_set in enumerate(i_variable.parents_set_candidates):
        #print "    Parents: ",[o.name for o in j_parents_set.variables]
        indexes_list=[]
        for k,k_combination in enumerate(j_parents_set.pruned_combinations):
            print "        Combination ",k+1, "from parents set", [o.name for o in j_parents_set.variables]," -> ", k_combination.expression , " == ", k_combination.pattern
            for l,l_rule in enumerate(k_combination.rules):
                print "            ",k_combination.probabilities_list[l],":",l_rule.head, ":-",l_rule.body," -> Representation=",l_rule.representation

            #Calculate occurences
            aux_configurations=copy.deepcopy(examples_configurations)
            for l,aux_configuration in enumerate(aux_configurations):
                head_configuration=aux_configuration[i]
                del aux_configuration[i]
                aux_configuration.append(head_configuration)

            for l,aux_configuration in enumerate(aux_configurations):
                new_configuration=[]
                for aux_item in aux_configuration:
                    if aux_item==True:
                        new_configuration.append(1)
                    else:
                        new_configuration.append(-1)
                aux_configurations[l]=new_configuration

            k_combination.occurences=[]

            indexes=[]
            for l, l_variable in enumerate(observed_variables):
                for m,parent_variable in enumerate(j_parents_set.variables):
                    if parent_variable.name==l_variable.name and l_variable.name!="":
                        indexes.append(l)
            for l,l_variable in enumerate(observed_variables):
                if j_parents_set.variable.name==l_variable.name:
                    indexes.append(l)

            k_combination.occurences=[0]*len(k_combination.configurations)
            for l,l_configuration in enumerate(k_combination.configurations):
                for m,m_configuration in enumerate(examples_configurations):
                    same_configuration=True
                    for n,index in enumerate(indexes):
                        if l_configuration[n]==1 and m_configuration[index]==False:
                            same_configuration=False
                            break
                        if l_configuration[n]==-1 and m_configuration[index]==True:
                            same_configuration=False
                            break
                    if same_configuration:
                        k_combination.occurences[l]+=examples_occurence[m]

            # print "                examples_configurations=",examples_configurations
            # print "                examples_occurence=",examples_occurence

            # print "                k_combination.configurations=",k_combination.configurations
            # print "                k_combination.occurences=",k_combination.occurences
            # print "                k_combination.body_configurations=",k_combination.body_configurations

            k_combination.calculate_multipliers()

            for l, l_configuration in enumerate(k_combination.configurations):
                rule_activated=False
                for m,m_rule in enumerate(k_combination.rules):
                    if k_combination.multipliers[l][m]!=0:
                        rule_activated=True
                        break
                if not rule_activated:
                    indexes_list.append(k)
                    break

        #print len(indexes_list)
        for k in sorted(indexes_list, reverse=True):
            del j_parents_set.pruned_combinations[k]

#Print combinations left
for i, i_variable in enumerate(observed_variables):
    print "\nVariable: ", i_variable.name
    for j, j_parents_set in enumerate(i_variable.parents_set_candidates):
        print "    Parents: ",[o.name for o in j_parents_set.variables]
        for k,k_combination in enumerate(j_parents_set.pruned_combinations):
            print "        Combination ",k+1, "from parents set", [o.name for o in j_parents_set.variables]," -> ", k_combination.expression , " == ", k_combination.pattern
            for l_rule in k_combination.rules: print "        ",l_rule.head, ":-",l_rule.body

print "\n###########################################"
print "#Estimate parameters for combinations left#"
print "###########################################"

for i, i_variable in enumerate(observed_variables):
    print "\nVariable: ", i_variable.name
    for j, j_parents_set in enumerate(i_variable.parents_set_candidates):
        print "    Parents: ",[o.name for o in j_parents_set.variables]
        for k,k_combination in enumerate(j_parents_set.pruned_combinations):
            print "        Combination ",k+1, "from parents set", [o.name for o in j_parents_set.variables]," -> ", k_combination.expression , " == ", k_combination.pattern
            for l,l_rule in enumerate(k_combination.rules):
                print "            ",k_combination.probabilities_list[l],":",l_rule.head, ":-",l_rule.body," -> Representation=",l_rule.representation

            #Calculate parameters
            k_combination.sollution()
            k_combination.score=0.0
            k_combination.score=-1*k_combination.minus_log_likelihood(k_combination.probabilities_list)

            print "            Log-likelihood=",k_combination.score
            k_combination.score-=k_combination.description_length
            print "            Description length=",k_combination.description_length
            print "            Score=",k_combination.score
            for l,l_rule in enumerate(k_combination.rules):
                print "           ",k_combination.probabilities_list[l],":",l_rule.head, ":-",l_rule.body," -> Representation=",l_rule.representation
            # for l, l_configuration in enumerate(k_combination.configurations):
            #     print "                ",l_configuration, " -> N=",k_combination.occurences[l]

print "\n#########################################"
print "#Find best combination for each parent set#"
print "###########################################"

for i_variable in observed_variables:
    for j_parents_set in i_variable.parents_set_candidates:
        best_score=j_parents_set.pruned_combinations[0].score
        best_combination=j_parents_set.pruned_combinations[0]
        for k, k_combination in enumerate(j_parents_set.pruned_combinations):
            if k_combination.score>best_score:
                best_score=k_combination.score
                best_combination=k_combination
        j_parents_set.best_combination=best_combination
        j_parents_set.score=best_score

for i, i_variable in enumerate(observed_variables):
    print "Variable: ", i_variable.name
    for j, j_parents_set in enumerate(i_variable.parents_set_candidates):
        print "    Parents: ",[o.name for o in j_parents_set.variables]
        print "        Best combination -> ", j_parents_set.best_combination.expression, " == ", j_parents_set.best_combination.pattern, ", description length=",j_parents_set.best_combination.description_length
        for l,l_rule in enumerate(j_parents_set.best_combination.rules):
            print "            ",j_parents_set.best_combination.probabilities_list[l],":",l_rule.head, ":-",l_rule.body," -> Representation=",l_rule.representation
        #for l, l_configuration in enumerate(k_combination.configurations):
            #print "                ",l_configuration, " -> N=",k_combination.occurences[l]

print"############################"
print"#Create CP_Bayes input file#"
print"############################"

fileToWrite = open(sys.argv[3], 'w' )

textToWrite=str(len(observed_variables))+"\n"
fileToWrite.write(textToWrite)

for i,i_variable in enumerate(observed_variables):

    #Print parents_set header
    textToWrite=str(i_variable.nickname) + " " + str(len(i_variable.parents_set_candidates)) + "\n"
    fileToWrite.write(textToWrite)
    print textToWrite

    #Create ordered_list
    ordered_list=[]
    # ordered_list.append(aux_parents_set)

    for j, j_parents_set in enumerate(i_variable.parents_set_candidates):
        found=False
        for k, ordered_parents_set in enumerate(ordered_list):
            if j_parents_set.score<=ordered_parents_set.score:
                ordered_list.insert(k,j_parents_set)
                found=True
                break
        if found==False:
            ordered_list.append(j_parents_set)
    #print "ordered_list = ", [p.name for p in o.variables for o in ordered_list]

    #Print ordered list
    for j_parents_set in ordered_list:
        parents_name=""
        aux=[]
        for parents_variable in j_parents_set.variables:
            parents_name=parents_name + " " + str(parents_variable.nickname)
            aux.append(parents_name)

        textToWrite=str(j_parents_set.score) + " " + str(len(aux)) + " " + parents_name +"\n"
        fileToWrite.write(textToWrite)
        print textToWrite

fileToWrite.close()

print"\n################################"
print"#Calculate score on testing set#"
print"################################"

print "\n########################"
print "#Store dataset examples#"
print "########################"

file_name=str(sys.argv[2])
with codecs.open(file_name, "r", encoding="utf-8-sig") as f:

    examples=[]
    for i_line in f:
        i_example=i_line.split(",")
        print "i_example=", i_example
        new_example=example()
        for j_observation in i_example:
            new_observation=observation()

            if "/+" in j_observation:
                new_observation.value=False
                aux_observation=str(j_observation[2:])
                aux_observation=aux_observation.replace("\r", "")
                aux_observation=aux_observation.replace("\n", "")
            else:
                new_observation.value=True
                aux_observation=str(j_observation)
                aux_observation=aux_observation.replace("\r", "")
                aux_observation=aux_observation.replace("\n", "")

            for k,k_variable in enumerate(observed_variables):
                if k_variable.name==aux_observation:
                    new_observation.variable=k_variable
                    break

            new_example.observations.append(new_observation)

        examples.append(new_example)

print "\n###############################"
print "#Build examples configurations#"
print "###############################"

examples_configurations=[]
examples_occurence=[]

for i,i_example in enumerate(examples):
    new_configuration=[]
    for j,j_observation in enumerate(i_example.observations):
        if j_observation.value==True:
            new_configuration.append(1)
        if j_observation.value==False:
            new_configuration.append(0)

    found_configuration=False
    for j, j_configuration in enumerate(examples_configurations):
        if j_configuration==new_configuration:
            found_configuration=True
            break
    if not found_configuration:
        examples_configurations.append(new_configuration)
        examples_occurence.append(1)
    else:
        examples_occurence[j]+=1

for i,i_example_configuration in enumerate(examples_configurations):
    print i+1,":",i_example_configuration, " -> N=",examples_occurence[i]

print "\n########################"
print "#Enter best parent sets#"
print "########################"

while True:
    for i, i_variable in enumerate(observed_variables):
        print "\nVariable: ", i_variable.name

        while True:
            user_input = raw_input("Enter parents for variable "+i_variable.name+":")
            parents_names = user_input.split()

            for j, j_parents_set in enumerate(i_variable.parents_set_candidates):
                print set([o.name for o in j_parents_set.variables])
                if set([o.name for o in j_parents_set.variables])==set(parents_names):
                    i_variable.parents_set=j_parents_set
                    for l,l_rule in enumerate(i_variable.parents_set.best_combination.rules):
                        print "           ",i_variable.parents_set.best_combination.probabilities_list[l],":",l_rule.head, ":-",l_rule.body," -> Representation=",l_rule.representation
                    break
            user_input = raw_input("Are you sure about the parents given?")
            if user_input == "y":
                break

    user_input = raw_input("Are you sure about the network built?")
    if user_input == "y":
        break

print "\n######################"
print "#Calculate occurences#"
print "######################"

for i, i_variable in enumerate(observed_variables):
    #print "\nVariable: ", i_variable.name
    j_parents_set=i_variable.parents_set
    #print "    Parents: ",[o.name for o in j_parents_set.variables]
    indexes_list=[]
    k_combination=j_parents_set.best_combination
    print "        Combination ",k+1, "from parents set", [o.name for o in j_parents_set.variables]," -> ", k_combination.expression , " == ", k_combination.pattern
    for l,l_rule in enumerate(k_combination.rules):
        print "            ",k_combination.probabilities_list[l],":",l_rule.head, ":-",l_rule.body," -> Representation=",l_rule.representation

    #Calculate occurences
    aux_configurations=copy.deepcopy(examples_configurations)
    for l,aux_configuration in enumerate(aux_configurations):
        head_configuration=aux_configuration[i]
        del aux_configuration[i]
        aux_configuration.append(head_configuration)

    for l,aux_configuration in enumerate(aux_configurations):
        new_configuration=[]
        for aux_item in aux_configuration:
            if aux_item==True:
                new_configuration.append(1)
            else:
                new_configuration.append(-1)
        aux_configurations[l]=new_configuration

    k_combination.occurences=[]

    indexes=[]
    for l, l_variable in enumerate(observed_variables):
        for m,parent_variable in enumerate(j_parents_set.variables):
            if parent_variable.name==l_variable.name and l_variable.name!="":
                indexes.append(l)
    for l,l_variable in enumerate(observed_variables):
        if j_parents_set.variable.name==l_variable.name:
            indexes.append(l)

    # print "indexes=",indexes
    # print "len(examples_configurations)=", len(examples_configurations)
    # print "len(k_combination.configurations)=",len(k_combination.configurations)

    k_combination.occurences=[0]*len(k_combination.configurations)
    for l,l_configuration in enumerate(k_combination.configurations):
        for m,m_configuration in enumerate(examples_configurations):
            same_configuration=True
            for n,index in enumerate(indexes):
                # print "n=",n
                # print "index=", index
                # print "len(l_configuration)=", len(l_configuration)
                # print "len(m_configuration)=", len(m_configuration)
                if l_configuration[n]==1 and m_configuration[index]==False:
                    same_configuration=False
                    break
                if l_configuration[n]==-1 and m_configuration[index]==True:
                    same_configuration=False
                    break
            if same_configuration:
                k_combination.occurences[l]+=examples_occurence[m]

    # print "                examples_configurations=",examples_configurations
    # print "                examples_occurence=",examples_occurence

    # print "                k_combination.configurations=",k_combination.configurations
    # print "                k_combination.occurences=",k_combination.occurences
    # print "                k_combination.body_configurations=",k_combination.body_configurations

    k_combination.calculate_multipliers()

    for l, l_configuration in enumerate(k_combination.configurations):
        rule_activated=False
        for m,m_rule in enumerate(k_combination.rules):
            if k_combination.multipliers[l][m]!=0:
                rule_activated=True
                break
        if not rule_activated:
            indexes_list.append(k)
            break

#Print combinations left
for i, i_variable in enumerate(observed_variables):
    print "\nVariable: ", i_variable.name
    j_parents_set=i_variable.parents_set
    print "    Parents: ",[o.name for o in j_parents_set.variables]
    k_combination=j_parents_set.best_combination
    print "        Best combination from parents set", [o.name for o in j_parents_set.variables]," -> ", k_combination.expression , " == ", k_combination.pattern
    for l_rule in k_combination.rules: print "        ",l_rule.head, ":-",l_rule.body

print "\n################################"
print "#Calculate score on testing_set#"
print "################################"

global_score=0
global_likelihood=0
parameters=0
for i, i_variable in enumerate(observed_variables):
    print "\nVariable: ", i_variable.name
    j_parents_set=i_variable.parents_set
    k_combination=i_variable.parents_set.best_combination

    print "        Best combination from parents set", [o.name for o in j_parents_set.variables]," -> ", k_combination.expression , " == ", k_combination.pattern
    for l,l_rule in enumerate(k_combination.rules):
        print "            ",k_combination.probabilities_list[l],":",l_rule.head, ":-",l_rule.body," -> Representation=",l_rule.representation

        # print "                examples_configurations=",examples_configurations
        # print "                examples_occurence=",examples_occurence

        # print "                k_combination.configurations=",k_combination.configurations
        # print "                k_combination.occurences=",k_combination.occurences
        # print "                k_combination.body_configurations=",k_combination.body_configurations

    k_combination.score=0.0
    k_combination.score=-1*k_combination.minus_log_likelihood(k_combination.probabilities_list)
    cutoff=0.5
    k_combination.accuracy=k_combination.accuracy(k_combination.probabilities_list,cutoff)

    #Compute likelihood
    print "            Log-likelihood=",k_combination.score
    global_likelihood+=k_combination.score

    #Compute penalization
    k_combination.score-=k_combination.description_length
    print "            Description length=",k_combination.description_length
    print "            Score=",k_combination.score

    #Compute final score
    global_score+=k_combination.score
    global_accuracy+=k_combination.accuracy

    # for l,l_rule in enumerate(k_combination.rules):
    #     print "           ",k_combination.probabilities_list[l],":",l_rule.head, ":-",l_rule.body," -> Representation=",l_rule.representation
    # for l, l_configuration in enumerate(k_combination.configurations):
    #     print "                ",l_configuration, " -> N=",k_combination.occurences[l]

    parameters+=len(k_combination.rules)

global_accuracy=global_accuracy/len(observed_variables)

print "Global score on testing set: ", global_score
print "Global likelihood on testing set: ", global_likelihood
print "Global accuracy on testing set: ", global_accuracy
print "Parameters: ", parameters
