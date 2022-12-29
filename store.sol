// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

contract Store{
    // assighn the contract creator as the store manager
    mapping(address => bool) admins;
    address payable Owner;
    address creator;

    constructor() public {
        Owner = payable(msg.sender); 
        creator = msg.sender; 
        admins[msg.sender] = true;
    }



    // access modifiers:
     modifier manager_access(){
        require(msg.sender == creator);
        _;
    }

    modifier admin_access(){
        // checks to see if the senders details are in the admin hashdict
        require(admins[msg.sender] == true);

        _;
    }

    // functions for the owner to add/remove admins
    function addAdmin(address new_admin) public manager_access(){
        admins[new_admin] = true;
    } 
    function removeAdmin(address new_admin) public manager_access(){
        admins[new_admin] = false;
    } 


    // item availability status
    enum i_availability{
        available, unavailable, no_stock
    }
    
    // item deslaration
    struct Item{
        string id;
        string name;
        i_availability availability;
        string store;
        uint value;
        uint quantity;
    }

    // hashdict for current items
    mapping(bytes32 => Item) Stock;

    function transferIn(uint _amount) public payable{ 
        Owner.transfer(_amount);
    }
  
}