

class Vehicle:
    def __init__(self, raw, previous_period):
        self.raw = raw
        self.previous_period = previous_period
        self.pretax_income = None

        # Calculate the following variables for the given period of the given product
        self.sales_price = self.raw['sales_price_per_unit']
        self.sales_price_adjustment = self.raw['sales_price_adjustment_percent']
        self.sales_volume = self.raw['sales_volume']
        self.sales_incentive = self.raw['sales_incentive']
        self.destination_charge = self.raw['destination_charge_per_unit']
        #
        self.direct_materials = self.raw['direct_materials']
        self.direct_labor = self.raw['direct_labor']
        self.overhead = self.raw['overhead']
        self.freight = self.raw['freight']
        self.duties_and_taxes = self.raw['duties_and_taxes']
        self.warranty_reserves = self.raw['warranty_reserve_percent_of_sales']
        #
        self.sales_and_marketing = self.raw['sales_and_marketing']
        self.operating_expense = self.raw['operating_expense']
        self.research_and_development = self.raw['research_and_development']
        self.salaries_and_wages = self.raw['salaries_and_wages']
        self.depreciation = self.raw['depreciation_expense']
        #
        self.receivable_collection_rate = self.raw['receivable_collection_rate']
        #
        self.state_tax = self.raw['state_tax']
        self.federal_tax = self.raw['federal_tax'],
        self.initial_investment = self.raw['initial_investment']
        #
        self.unit_sales = None
        self.incentives = None
        self.destination = None
        self.gross_sales = None

        self.dm = None
        self.dl = None
        self.oh = None
        self.fh = None
        self.dt = None
        self.warranty = None
        self.cost_of_sales = None

        self.operating_expenses = None
        self.pretax_income = None
        self.net_collected_revenues = None
        self.warranty_costs = None
        self.warranty_reserve_cost = None

        # Cost of sales
        self.standard_cost = None
        self.net_cash_production_costs = None
        self.gross_cash_operating_expenses = None
        self.net_cash_non_production_costs = None
        self.net_non_operating_cash_flow = None
        self.cash_paid_state_taxes = None
        self.cash_paid_federal_taxes = None

    def reusable_variables(self):
        # Gross Sales
        ## 1. Unit Sales
        self.unit_sales = self.sales_volume * (self.sales_price * (1 + self.sales_price_adjustment))
        ## 2. Sales Incentives
        self.incentives = self.sales_volume * self.sales_incentive
        ## 3. Sales destination charges
        self.destination = self.sales_volume * self.destination_charge
        ## Total Gross Sales
        self.gross_sales = self.unit_sales + self.incentives + self.destination

        # Cost of Sales
        self.dm = self.sales_volume * self.direct_materials
        self.dl = self.sales_volume * self.direct_labor
        self.oh = self.sales_volume * self.overhead
        self.fh = self.sales_volume * self.freight
        self.dt = self.sales_volume * self.duties_and_taxes
        self. warranty = self.gross_sales * self.warranty_reserves
        self.cost_of_sales = -(self.dm + self.dl + self.oh + self.fh + self.dt + self.warranty)

        # Gross Profit
        self.gross_profit = self.gross_sales + self.cost_of_sales

        # Operating Expenses
        self.operating_expenses = self.sales_and_marketing + self.operating_expense + self.research_and_development + self.salaries_and_wages + self.depreciation

        # Pretax Income
        self.pretax_income = self.gross_profit - self.operating_expenses

        # Net Collected Revenues
        self.net_collected_revenues = self.gross_sales * self.receivable_collection_rate
        # Warranty Reserves
        self.warranty_costs = -(self.unit_sales * self.raw['warranty_reserve_percent_of_sales'])
        self.warranty_reserve_cost = (self.gross_sales + ((self.sales_incentive + self.destination_charge) * self.sales_volume) + (self.sales_price * (1 + self.sales_price_adjustment)) * self.sales_volume) * self.warranty_costs

        # Cost of sales
        self.standard_cost = - (self.dm + self.dl + self.oh + self.fh + self.dt)
        self.net_cash_production_costs = self.standard_cost * self.raw['production_supplier_pay_rate']
        self.gross_cash_operating_expenses = sum((self.warranty_costs,
                                             self.sales_and_marketing,
                                             self.operating_expense,
                                             self.research_and_development,
                                             self.salaries_and_wages))

        self.net_cash_non_production_costs = self.gross_cash_operating_expenses * self.raw['non_production_supplier_pay_rate']
        self.net_non_operating_cash_flow, self.cash_paid_state_taxes, self.cash_paid_federal_taxes = self.non_operating_income_expense()

        reusable_dict = {
            'sales_price_per_unit': self.sales_price,
            'unit_sales': self.unit_sales,
            'destination_charge': self.destination_charge,
            'gross_sales': self.gross_sales,
            'net_collected_revenues': self.net_collected_revenues,
            'materials_cost': self.dm,
            'labor_cost': self.dl,
            'overhead_cost': self.oh,
            'freight_cost': self.fh,
            'taxes_cost': self.dt,
            'standard_cost': self.standard_cost,
            'net_cash_production_costs': self.net_cash_production_costs,
            'warranty_costs': self.warranty_costs,
            'gross_cash_operating_expenses': self.gross_cash_operating_expenses,
            'net_cash_non_production_costs': self.net_cash_non_production_costs,
            'direct_materials': self.direct_materials,
            'direct_labor': self.direct_labor,
            'overhead': self.overhead,
            'freight': self.freight,
            'duties_and_taxes': self.duties_and_taxes,
            'cash_paid_state_taxes': self.cash_paid_state_taxes,
            'cash_paid_federal_taxes': self.cash_paid_federal_taxes,
            'sales_volume': self.sales_volume,
            'state_tax': self.state_tax,
            'federal_tax': self.federal_tax,
            'initial_investment': self.initial_investment
        }

        return reusable_dict


    def cash_collections(self):
        reusable_dict = self.reusable_variables()
        if self.previous_period is not None:
            post_period_collections_cash = self.previous_period['gross_sales'] - \
                                           self.previous_period['net_collected_revenues']
        else:
            post_period_collections_cash = 0

        total_cash_collections = reusable_dict['net_collected_revenues'] + post_period_collections_cash
        return total_cash_collections


    def production_disbursements(self):
        reusable_dict = self.reusable_variables()

        if self.previous_period is not None:
            post_period_payments_production = -(self.previous_period['net_cash_production_costs']) + \
                                              self.previous_period['standard_cost']
        else:
            post_period_payments_production = 0

        total_supplier_production_payments = reusable_dict['net_cash_production_costs'] + post_period_payments_production
        return total_supplier_production_payments


    def non_production_disbursements(self):
        reusable_dict = self.reusable_variables()

        if self.previous_period is not None:
            post_period_payments_non_production = -(self.previous_period['net_cash_non_production_costs']) \
                                                  + self.previous_period['gross_cash_operating_expenses']
        else:
            post_period_payments_non_production = 0

        total_non_production_supplier_payments = reusable_dict['net_cash_non_production_costs'] + post_period_payments_non_production
        return total_non_production_supplier_payments


    def non_operating_income_expense(self):
        if self.previous_period is not None:
            try:
                #                       -(C11                                                   - ((C157                                 * C161                                    ) + (C157                                 * C162                                ) + (C157                                 * C163                            ) + (C157                                 * C164                           ) + ((C157                                 * C161                                    ) * C165                                    )) + C33                                                  ) * C184
                cash_paid_state_taxes = -(self.previous_period['unit_sales'] - ((self.previous_period['sales_volume'] * self.previous_period['direct_materials']) + (self.previous_period['sales_volume'] * self.previous_period['direct_labor']) + (self.previous_period['sales_volume'] * self.previous_period['overhead']) + (self.previous_period['sales_volume'] * self.previous_period['freight']) + ((self.previous_period['sales_volume'] * self.previous_period['direct_materials']) * self.previous_period['duties_and_taxes'])) + self.previous_period['gross_cash_operating_expenses']) * self.previous_period['state_tax']
            except:
                cash_paid_state_taxes = 0
            try:
                #                         -(C11                                + ((C21                                   / C20                                              ) * C157                                ) + C33                                                   + C41                                          ) * C185
                cash_paid_federal_taxes = -(self.previous_period['unit_sales'] + ((self.previous_period['standard_cost'] / self.previous_period['planned_production_volume']) * self.previous_period['sales_volume']) + self.previous_period['gross_cash_operating_expenses'] + self.previous_period['cash_paid_state_taxes']) * self.previous_period['federal_tax']
            except:
                cash_paid_federal_taxes = 0

        else:
            cash_paid_state_taxes = 0
            cash_paid_federal_taxes = 0


        net_non_operating_cash_flow = self.raw['investment_earnings'] + cash_paid_state_taxes + cash_paid_federal_taxes
        return net_non_operating_cash_flow, cash_paid_state_taxes, cash_paid_federal_taxes


    def financing_activities(self):
        net_financing_activities = self.raw['debt_issuance'] + \
                                   self.raw['debt_repayments'] + \
                                   self.raw['acquisition_expenditures'] + \
                                   self.raw['capital_expenditures']
        return net_financing_activities


    def calculate_net_cash_flow(self):
        total_cash_collections = self.cash_collections()
        total_supplier_production_payments = self.production_disbursements()
        total_non_production_supplier_payments = self.non_production_disbursements()
        net_non_operating_cash_flow, cash_paid_state_taxes, cash_paid_federal_taxes = self.non_operating_income_expense()
        net_financing_activities = self.financing_activities()

        ncf = sum((total_cash_collections,
                   total_supplier_production_payments,
                   total_non_production_supplier_payments,
                   net_non_operating_cash_flow,
                   net_financing_activities))
        return ncf