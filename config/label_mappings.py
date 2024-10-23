def get_label_mapping(label_type):
    if label_type == "product":
        return {
            0: "Credit Reporting",
            1: "Credit card",
            2: "Debt collection",
            3: "Checking/savings account",
            4: "Loans / Mortgage"
        }
    elif label_type == "sub_product":
        return {
            0: "Credit reporting",
            1: "Store credit card",
            2: "General-purpose credit card or charge card",
            3: "Credit card debt",
            4: "Checking account",
            5: "Other personal consumer report",
            6: "Auto debt",
            7: "Federal student loan servicing",
            8: "Other debt",
            9: "Telecommunications debt",
            10: "Medical debt",
            11: "Conventional home mortgage",
            12: "Other banking product or service",
            13: "Rental debt",
            14: "Savings account"
        }
    elif label_type == "issue":
        return {
            0: "Incorrect information on your report",
            1: "Problem when making payments",
            2: "Getting a credit card",
            3: "Problem with a company's investigation into an existing problem",
            4: "Attempts to collect debt not owed",
            5: "Managing an account",
            6: "Improper use of your report",
            7: "Problem with a purchase shown on your statement",
            8: "False statements or representation",
            9: "Dealing with your lender or servicer",
            10: "Written notification about debt",
            11: "Trouble during payment process",
            12: "Fees or interest",
            13: "Problem with a lender or other company charging your account",
            14: "Took or threatened to take negative or legal action",
            15: "Other features, terms, or problems"
        }
    elif label_type == "sub_issue":
        return {
            0: 'Information belongs to someone else',
            1: 'Account status incorrect',
            2: 'Their investigation did not fix an error on your report',
            3: "Credit inquiries on your report that you don't recognize",
            4: 'Reporting company used your report improperly',
            5: 'Account information incorrect',
            6: 'Was not notified of investigation status or results',
            7: 'Problem during payment process',
            8: 'Card opened without my consent or knowledge',
            9: 'Debt is not yours',
            10: 'Deposits and withdrawals',
            11: 'Card was charged for something you did not purchase with the card',
            12: 'Attempted to collect wrong amount',
            13: 'Trouble with how payments are being handled',
            14: 'Debt was result of identity theft',
            15: "Didn't receive notice of right to dispute",
            16: 'Problem using a debit or ATM card',
            17: 'Escrow, taxes, or insurance',
            18: 'Problem with fees',
            19: "Credit card company isn't resolving a dispute about a purchase on your statement",
            20: 'Transaction was not authorized',
            21: 'Debt was paid',
            22: 'Threatened or suggested your credit would be damaged',
            23: 'Other problem',
            24: "Didn't receive enough information to verify debt",
            25: 'Funds not handled or disbursed as instructed'
        }