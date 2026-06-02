class Calculator {
    constructor(previousOperandTextElement, currentOperandTextElement) {
        this.previousOperandTextElement = previousOperandTextElement;
        this.currentOperandTextElement = currentOperandTextElement;
        this.clear();
    }

    clear() {
        this.currentOperand = '0';
        this.previousOperand = '';
        this.operation = undefined;
    }

    delete() {
        if (this.currentOperand === '0') return;
        this.currentOperand = this.currentOperand.toString().slice(0, -1);
        if (this.currentOperand === '') {
            this.currentOperand = '0';
        }
    }

    appendNumber(number) {
        if (number === '.' && this.currentOperand.includes('.')) return;
        if (this.currentOperand === '0' && number !== '.') {
            this.currentOperand = number.toString();
        } else {
            this.currentOperand = this.currentOperand.toString() + number.toString();
        }
    }

    chooseOperation(operation) {
        if (this.currentOperand === '' && this.previousOperand === '') return;
        
        // If we are choosing a new operation but haven't entered a new number yet,
        // just update the operation symbol.
        if (this.currentOperand === '') {
            this.operation = operation;
            return;
        }

        if (this.previousOperand !== '') {
            this.compute();
        }

        // Apply percentage immediately if selected
        if (operation === '%') {
            const current = parseFloat(this.currentOperand);
            if (isNaN(current)) return;
            this.currentOperand = (current / 100).toString();
            this.operation = undefined;
            this.previousOperand = '';
            return;
        }

        this.operation = operation;
        this.previousOperand = this.currentOperand;
        this.currentOperand = '';
    }

    compute() {
        let computation;
        const prev = parseFloat(this.previousOperand);
        const current = parseFloat(this.currentOperand);
        
        if (isNaN(prev) || isNaN(current)) return;
        
        switch (this.operation) {
            case '+':
                computation = prev + current;
                break;
            case '-':
                computation = prev - current;
                break;
            case '×':
                computation = prev * current;
                break;
            case '÷':
                if (current === 0) {
                    computation = 'Error';
                } else {
                    computation = prev / current;
                }
                break;
            default:
                return;
        }
        
        this.currentOperand = computation;
        this.operation = undefined;
        this.previousOperand = '';
    }

    getDisplayNumber(number) {
        if (number === 'Error') return 'Error';
        
        const stringNumber = number.toString();
        const integerDigits = parseFloat(stringNumber.split('.')[0]);
        const decimalDigits = stringNumber.split('.')[1];
        let integerDisplay;
        
        if (isNaN(integerDigits)) {
            integerDisplay = '';
        } else {
            integerDisplay = integerDigits.toLocaleString('en', {
                maximumFractionDigits: 0
            });
        }
        
        if (decimalDigits != null) {
            return `${integerDisplay}.${decimalDigits}`;
        } else {
            return integerDisplay;
        }
    }

    updateDisplay() {
        if (this.currentOperand === 'Error') {
            this.currentOperandTextElement.innerText = 'Error';
            this.previousOperandTextElement.innerText = '';
            return;
        }

        this.currentOperandTextElement.innerText = this.getDisplayNumber(this.currentOperand);
        
        if (this.operation != null) {
            this.previousOperandTextElement.innerText = 
                `${this.getDisplayNumber(this.previousOperand)} ${this.operation}`;
        } else {
            this.previousOperandTextElement.innerText = '';
        }

        // Adjust font size dynamically based on length
        this.adjustFontSize();
    }

    adjustFontSize() {
        const length = this.currentOperandTextElement.innerText.length;
        if (length > 12) {
            this.currentOperandTextElement.style.fontSize = '1.5rem';
        } else if (length > 8) {
            this.currentOperandTextElement.style.fontSize = '2rem';
        } else {
            this.currentOperandTextElement.style.fontSize = '2.5rem';
        }
    }
}

const numberButtons = document.querySelectorAll('[data-number]');
const operationButtons = document.querySelectorAll('[data-operator]');
const equalsButton = document.querySelector('[data-action="equals"]');
const deleteButton = document.querySelector('[data-action="delete"]');
const clearButton = document.querySelector('[data-action="clear"]');
const previousOperandTextElement = document.getElementById('previous-operand');
const currentOperandTextElement = document.getElementById('current-operand');

const calculator = new Calculator(previousOperandTextElement, currentOperandTextElement);

numberButtons.forEach(button => {
    button.addEventListener('click', () => {
        // Prevent typing after an Error
        if (calculator.currentOperand === 'Error') {
            calculator.clear();
        }
        calculator.appendNumber(button.dataset.number);
        calculator.updateDisplay();
        
        // Add subtle click animation class
        button.classList.add('clicked');
        setTimeout(() => button.classList.remove('clicked'), 100);
    });
});

operationButtons.forEach(button => {
    button.addEventListener('click', () => {
        if (calculator.currentOperand === 'Error') return;
        calculator.chooseOperation(button.dataset.operator);
        calculator.updateDisplay();
    });
});

equalsButton.addEventListener('click', button => {
    if (calculator.currentOperand === 'Error') return;
    calculator.compute();
    calculator.updateDisplay();
});

clearButton.addEventListener('click', () => {
    calculator.clear();
    calculator.updateDisplay();
});

deleteButton.addEventListener('click', () => {
    if (calculator.currentOperand === 'Error') {
        calculator.clear();
    } else {
        calculator.delete();
    }
    calculator.updateDisplay();
});

// Keyboard support
document.addEventListener('keydown', e => {
    if (calculator.currentOperand === 'Error' && e.key !== 'Escape') return;

    if (e.key >= 0 && e.key <= 9 || e.key === '.') {
        calculator.appendNumber(e.key);
        calculator.updateDisplay();
    }
    if (e.key === '=' || e.key === 'Enter') {
        e.preventDefault(); // Prevent form submission if any
        calculator.compute();
        calculator.updateDisplay();
    }
    if (e.key === 'Backspace') {
        calculator.delete();
        calculator.updateDisplay();
    }
    if (e.key === 'Escape') {
        calculator.clear();
        calculator.updateDisplay();
    }
    if (e.key === '+' || e.key === '-') {
        calculator.chooseOperation(e.key);
        calculator.updateDisplay();
    }
    if (e.key === '*') {
        calculator.chooseOperation('×');
        calculator.updateDisplay();
    }
    if (e.key === '/') {
        e.preventDefault(); // Prevent quick search in Firefox
        calculator.chooseOperation('÷');
        calculator.updateDisplay();
    }
    if (e.key === '%') {
        calculator.chooseOperation('%');
        calculator.updateDisplay();
    }
});
