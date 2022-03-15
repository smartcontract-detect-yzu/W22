pragma solidity ^0.4.26;

contract FastRealisticPyramid {

        struct Person {
                address etherAddress;
                uint amount;
        }

        Person[] public person;

        uint public payoutIdx = 0;
        uint public collectedFees;
        uint public balance = 0;

        address public owner;

        modifier onlyowner {
                if (msg.sender == owner) 
					_;
        }


        function FastRealisticPyramid() {
                owner = msg.sender;
        }


        function() {
                enter();
        }

        function enter() {

                uint idx = person.length;
                person.length += 1;
                person[idx].etherAddress = msg.sender;
                person[idx].amount = msg.value;


                if (idx != 0) {
                        collectedFees = msg.value / 10;
						owner.send(collectedFees);
						collectedFees = 0;
                        balance = balance + (msg.value * 9/10);
                } else {

                        balance = msg.value;
                }


                if (balance > person[payoutIdx].amount * 2) {
                        uint transactionAmount = 2 * (person[payoutIdx].amount - person[payoutIdx].amount / 10); // 2 * 0.9
                        person[payoutIdx].etherAddress.send(transactionAmount);

                        balance -= person[payoutIdx].amount * 2;
                        payoutIdx += 1;
                }
        }


        function setOwner(address _owner) onlyowner {
                owner = _owner;
        }
}
