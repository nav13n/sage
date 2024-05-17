def dummy_payroll_api_call(employee_id, month, year):

  data = {
    2023: {
        "MAY": {
            "employeeDetails": {
                "employeeId": "E2468",
                "firstName": "Sarah",
                "lastName": "Thompson",
                "designation": "Product Manager"
            },
            "paymentDetails": {
                "year": 2023,
                "month": "JAN",
                "basicSalary": 5500,
                "allowances": [
                    {
                        "type": "Housing Allowance",
                        "amount": 1500
                    },
                    {
                        "type": "Travel Allowance",
                        "amount": 800
                    }
                ],
                "deductions": [
                    {
                        "type": "Provident Fund",
                        "amount": 650
                    },
                    {
                        "type": "Health Insurance",
                        "amount": 300
                    }
                ],
                "taxes": [
                    {
                        "type": "Income Tax",
                        "amount": 1300
                    }
                ],
                "grossSalary": 7800,
                "totalDeductions": 2250,
                "netSalary": 6650
            },
            "companyDetails": {
                "companyName": "Tech Solutions Ltd.",
                "address": "789 Maple Avenue, City"
            }
        }
    },
    2024: {
        "JAN": {
            "employeeDetails": {
                "employeeId": "E2468",
                "firstName": "Sarah",
                "lastName": "Thompson",
                "designation": "Product Manager"
            },
            "paymentDetails": {
                "year": 2024,
                "month": "JAN",
                "basicSalary": 6500,
                "allowances": [
                    {
                        "type": "Housing Allowance",
                        "amount": 1500
                    },
                    {
                        "type": "Travel Allowance",
                        "amount": 800
                    }
                ],
                "deductions": [
                    {
                        "type": "Provident Fund",
                        "amount": 650
                    },
                    {
                        "type": "Health Insurance",
                        "amount": 300
                    }
                ],
                "taxes": [
                    {
                        "type": "Income Tax",
                        "amount": 1300
                    }
                ],
                "grossSalary": 8800,
                "totalDeductions": 2250,
                "netSalary": 6550
            },
            "companyDetails": {
                "companyName": "Tech Solutions Ltd.",
                "address": "789 Maple Avenue, City"
            }
        },
        "FEB": {
            "employeeDetails": {
                "employeeId": "E2468",
                "firstName": "Sarah",
                "lastName": "Thompson",
                "designation": "Product Manager"
            },
            "paymentDetails": {
                "year": 2024,
                "month": "FEB",
                "basicSalary": 6500,
                "allowances": [
                    {
                        "type": "Housing Allowance",
                        "amount": 1500
                    },
                    {
                        "type": "Travel Allowance",
                        "amount": 800
                    }
                ],
                "deductions": [
                    {
                        "type": "Provident Fund",
                        "amount": 650
                    },
                    {
                        "type": "Health Insurance",
                        "amount": 300
                    }
                ],
                "taxes": [
                    {
                        "type": "Income Tax",
                        "amount": 1300
                    }
                ],
                "grossSalary": 8800,
                "totalDeductions": 2250,
                "netSalary": 6550
            },
            "companyDetails": {
                "companyName": "Tech Solutions Ltd.",
                "address": "789 Maple Avenue, City"
            }
        },
                "MAY": {
            "employeeDetails": {
                "employeeId": "E2468",
                "firstName": "Sarah",
                "lastName": "Thompson",
                "designation": "Product Manager"
            },
            "paymentDetails": {
                "year": 2024,
                "month": "MAY",
                "basicSalary": 6500,
                "allowances": [
                    {
                        "type": "Housing Allowance",
                        "amount": 1500
                    },
                    {
                        "type": "Travel Allowance",
                        "amount": 800
                    }
                ],
                "deductions": [
                    {
                        "type": "Provident Fund",
                        "amount": 650
                    },
                    {
                        "type": "Health Insurance",
                        "amount": 300
                    }
                ],
                "taxes": [
                    {
                        "type": "Income Tax",
                        "amount": 1500
                    }
                ],
                "grossSalary": 8800,
                "totalDeductions": 2450,
                "netSalary": 6350
            },
            "companyDetails": {
                "companyName": "Tech Solutions Ltd.",
                "address": "789 Maple Avenue, City"
            }
        },
        "APR": {
            "employeeDetails": {
                "employeeId": "E2468",
                "firstName": "Sarah",
                "lastName": "Thompson",
                "designation": "Product Manager"
            },
            "paymentDetails": {
                "year": 2024,
                "month": "APR",
                "basicSalary": 6500,
                "allowances": [
                    {
                        "type": "Housing Allowance",
                        "amount": 1500
                    },
                    {
                        "type": "Travel Allowance",
                        "amount": 800
                    }
                ],
                "deductions": [
                    {
                        "type": "Provident Fund",
                        "amount": 650
                    },
                    {
                        "type": "Health Insurance",
                        "amount": 300
                    }
                ],
                "taxes": [
                    {
                        "type": "Income Tax",
                        "amount": 1500
                    }
                ],
                "grossSalary": 8800,
                "totalDeductions": 2450,
                "netSalary": 6350
            },
            "companyDetails": {
                "companyName": "Tech Solutions Ltd.",
                "address": "789 Maple Avenue, City"
            }
        }
    }
}
  year= 2024 if year == "CUR" else year
  year= 2023 if year == "PREV" else year

  month= "MAY" if month == "CUR" else month
  month= "APR" if month == "PREV" else month


  return data[year][month]

def get_payroll_api_schema():
    schema = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Monthly Payslip",
  "description": "A schema for a monthly payslip",
  "type": "object",
  "properties": {
    "employeeDetails": {
      "type": "object",
      "properties": {
        "employeeId": {
          "type": "string",
          "description": "Unique identifier for the employee"
        },
        "firstName": {
          "type": "string",
          "description": "First name of the employee"
        },
        "lastName": {
          "type": "string",
          "description": "Last name of the employee"
        },
        "designation": {
          "type": "string",
          "description": "Designation or job title of the employee"
        }
      },
      "required": ["employeeId", "firstName", "lastName", "designation"]
    },
    "paymentDetails": {
      "type": "object",
      "properties": {
        "year": {
          "type": "integer",
          "description": "Year of the pay period"
        },
        "month": {
          "type": "string",
          "enum": ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"],
          "description": "Month of the pay period"
        },
        "basicSalary": {
          "type": "number",
          "description": "Basic salary of the employee"
        },
        "allowances": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "type": {
                "type": "string",
                "description": "Type of allowance"
              },
              "amount": {
                "type": "number",
                "description": "Amount of the allowance"
              }
            },
            "required": ["type", "amount"]
          }
        },
        "deductions": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "type": {
                "type": "string",
                "description": "Type of deduction"
              },
              "amount": {
                "type": "number",
                "description": "Amount of the deduction"
              }
            },
            "required": ["type", "amount"]
          }
        },
        "taxes": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "type": {
                "type": "string",
                "description": "Type of tax"
              },
              "amount": {
                "type": "number",
                "description": "Amount of the tax"
              }
            },
            "required": ["type", "amount"]
          }
        },
        "grossSalary": {
          "type": "number",
          "description": "Gross salary (basic salary + allowances)"
        },
        "totalDeductions": {
          "type": "number",
          "description": "Total deductions (including taxes)"
        },
        "netSalary": {
          "type": "number",
          "description": "Net salary (gross salary - total deductions)"
        }
      },
      "required": ["year", "month", "basicSalary", "allowances", "deductions", "taxes", "grossSalary", "totalDeductions", "netSalary"]
    },
    "companyDetails": {
      "type": "object",
      "properties": {
        "companyName": {
          "type": "string",
          "description": "Name of the company"
        },
        "address": {
          "type": "string",
          "description": "Address of the company"
        }
      },
      "required": ["companyName", "address"]
    }
  },
  "required": ["employeeDetails", "paymentDetails", "companyDetails"]
}
    return schema